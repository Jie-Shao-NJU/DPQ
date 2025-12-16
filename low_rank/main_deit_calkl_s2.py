# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import math
import random
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, DSDTraining

from models.deit_model import VisionTransformer
from models.deit_model_pca_adaptive import VisionTransformer as VisionTransformerAda

from tqdm import tqdm
from PIL import Image
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--block-i', type=int, default=-1)
    parser.add_argument('--fc-i', type=int, default=-1)
    parser.add_argument('--dim-i', type=int, default=-1)

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def merge_model(checkpoint, block_i, fc_i, dim_i, load_path):
    if fc_i == 1:
        fc_name = "fc1"
        layer_name = ".mlp."
    elif fc_i == 2:
        fc_name = "fc2"
        layer_name = ".mlp."

    elif fc_i == 3:
        fc_name = "qkv"
        layer_name = ".attn."

    u_name = load_path +  fc_name + "_" + str(block_i) + "_u.pt"
    v_name = load_path + fc_name + "_" + str(block_i) + "_v.pt"
    avg_name = load_path + fc_name + "_" + str(block_i) + "_avg.pt"

    u = torch.load(u_name, map_location='cpu').cpu()
    v = torch.load(v_name, map_location='cpu').cpu()
    avg = torch.load(avg_name, map_location='cpu').cpu()

    new_u = u[:,:dim_i]
    new_v = v[:dim_i,:]
    b = (torch.eye(new_u.shape[0]).cpu()- new_u @ new_v) @ avg

    original_w_name = "blocks." + str(block_i) + layer_name + fc_name + ".weight"
    original_b_name = "blocks." + str(block_i) + layer_name + fc_name + ".bias"
    original_w = checkpoint[original_w_name]
    original_b = checkpoint[original_b_name]

    new_w = new_v @ original_w
    new_b = original_b @ new_v.transpose(0,1)

    checkpoint[original_w_name] = new_w
    checkpoint[original_b_name] = new_b


    k_split = original_w_name.split(".")
    k_split[-2] = "uv" + str((fc_i- 1)%2 + 1)
    print(".".join(k_split))
    checkpoint[".".join(k_split)] = new_u

    k_split[-1] = "bias"
    checkpoint[".".join(k_split)] = b.reshape(-1)

    for k,v in checkpoint.items():
        print(k,v.shape)
    return checkpoint

def main(args, config):
    data_loader_val = torch.utils.data.DataLoader(
        datasets.ImageFolder("/mnt/ramdisk/ImageNet/fewshot2_train/", transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])),
        batch_size=config.DATA.BATCH_SIZE, shuffle=False,
        num_workers=6)
    
    fc1_len = [128,128, 128,128, 128,128, 128,128, 128,128, 128,128]
    fc2_len = [128,128, 128,128, 128,128, 128,128, 128,128, 128,128]
    qkv_len = [128,128, 128,128, 128,128, 128,128, 128,128, 128,128]
    proj_len = [128,128, 128,128, 128,128, 128,128, 128,128, 128,128]


    fc1_use_uv = [False, False, False, False, False, False, False, False,  False, False,  False, False, False]
    fc2_use_uv = [False, False, False, False, False, False, False, False,  False, False,  False, False, False]
    qkv_use_uv = [False, False, False, False, False, False, False, False,  False, False,  False, False, False]
    proj_use_uv = [False, False, False, False, False, False, False, False,  False, False,  False, False, False]

    if args.fc_i == 1:
        fc1_use_uv[args.block_i] = True
        fc1_len[args.block_i] = args.dim_i
    elif args.fc_i == 2:
        fc2_use_uv[args.block_i] = True
        fc2_len[args.block_i] = args.dim_i
    elif args.fc_i == 3:
        qkv_use_uv[args.block_i] = True
        qkv_len[args.block_i] = args.dim_i


    model = VisionTransformerAda(patch_size=16,  embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True,
            qkv_use_uv = qkv_use_uv,proj_use_uv=proj_use_uv,fc1_use_uv = fc1_use_uv,fc2_use_uv = fc2_use_uv,qkv_len = qkv_len,proj_len = proj_len, fc1_len = fc1_len, fc2_len = fc2_len)
    teacher_model = VisionTransformer(patch_size=16,  embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True)

    # model = VisionTransformerAda(patch_size=16,  embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True,
    #         qkv_use_uv = qkv_use_uv,proj_use_uv=proj_use_uv,fc1_use_uv = fc1_use_uv,fc2_use_uv = fc2_use_uv,qkv_len = qkv_len,proj_len = proj_len, fc1_len = fc1_len, fc2_len = fc2_len)
    # teacher_model = VisionTransformer(patch_size=16,  embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True)


    # model = VisionTransformerAda(patch_size=16,  embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
    #         qkv_use_uv = qkv_use_uv,fc1_use_uv = fc1_use_uv,fc2_use_uv = fc2_use_uv,qkv_len = qkv_len, fc1_len = fc1_len, fc2_len = fc2_len)
    # teacher_model = VisionTransformer(patch_size=16,  embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True)

    teacher_model_sum = sum(p.numel() for p in teacher_model.state_dict().values())
    model_sum = sum(p.numel() for p in model.state_dict().values())
    if model_sum > teacher_model_sum:
        return

    model.cuda()
    teacher_model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module
    teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    teacher_model_without_ddp = teacher_model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")


    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    # load_path = "deit/ffn_output/all_fc_ada/deit_b/deit_b_fewshot2_"
    # load_path = "deit/ffn_output/all_fc_ada/deit_s/deit_s_fewshot2_"
    load_path = "deit/ffn_output/all_fc_ada/deit_t/deit_t_fewshot2_"
    new_checkpoint = merge_model(checkpoint['model'], args.block_i, args.fc_i, args.dim_i, load_path)


    if 'model' in new_checkpoint.keys():
        msg = model_without_ddp.load_state_dict(new_checkpoint['model'], strict=False)
    else:
        msg = model_without_ddp.load_state_dict(new_checkpoint, strict=False)
    logger.info(msg)

    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    if 'model' in checkpoint.keys():
        msg = teacher_model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    else:
        msg = teacher_model_without_ddp.load_state_dict(checkpoint, strict=False)
    logger.info(msg)


    acc1, acc5, loss = validate(config, data_loader_val, model, teacher_model)
    logger.info(f"Accuracy of the network on the val test images: {acc1:.1f}%")


@torch.no_grad()
def validate(config, data_loader, model, teacher_model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    teacher_model.eval()


    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    kl = AverageMeter()

    end = time.time()


    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)


        output,_ = model(images)
        teacher_output,_ = teacher_model(images)

        logsoftmax = torch.nn.LogSoftmax(dim=1).cuda()
        softmax = torch.nn.Softmax(dim=1).cuda()
        distil_loss = torch.sum(
            torch.sum(softmax(teacher_output) * (logsoftmax(teacher_output)-logsoftmax(output)), dim=1))

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)
        distil_loss = reduce_tensor(distil_loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
        kl.update(distil_loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'kl {kl.val:.3f} ({kl.avg:.3f})\t'

                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}  KL {kl.avg:.3f}')

    f_name = "deit/" + str(args.block_i) + "_" + str(args.fc_i) + ".txt"
    with open(f_name, "a") as f:
        f.writelines(str(args.dim_i) + "," + str(kl.avg) + "\n")

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        # logger.info(f"Full config saved to {path}")

    # print config
    # logger.info(config.dump())

    main(args, config)
