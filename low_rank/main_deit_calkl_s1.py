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

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def main(config):
    data_loader_val = torch.utils.data.DataLoader(
        datasets.ImageFolder("/mnt/ramdisk/ImageNet/fewshot2_train/", transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])),
        batch_size=config.DATA.BATCH_SIZE, shuffle=False,
        num_workers=6)
    
    model = VisionTransformer(patch_size=16,  embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True)
    # model = VisionTransformer(patch_size=16,  embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True)
    # model = VisionTransformer(patch_size=16,  embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True)


    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")


    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    if 'model' in checkpoint.keys():
        msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    else:
        msg = model_without_ddp.load_state_dict(checkpoint, strict=False)

    logger.info(msg)
    acc1, acc5, loss = validate(config, data_loader_val, model)
    logger.info(f"Accuracy of the network on the val test images: {acc1:.1f}%")


def cal_cov(tensor_i, avg_tensor_i, ratio):
    u,s,vh = torch.linalg.svd(tensor_i, full_matrices=False)
    return  u,vh, avg_tensor_i

class output_hook(object):
    """
	Forward_hook used to get the output of the intermediate layer. 
	"""
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def clear(self):
        self.outputs = None

def cal_output_sum_square(fc_output_list, fc_square_list, fc_sum_list, fc_count_list):
    for i in range(len(fc_output_list)):
        tensor_i = fc_output_list[i].flatten(0,1)
        fc_sum = torch.sum(tensor_i, dim=0)
        fc_square = tensor_i.T @ tensor_i
        fc_count_list[i] += tensor_i.shape[0]
        if fc_square_list[i] is None:
            fc_sum_list[i] = fc_sum
            fc_square_list[i] = fc_square
        else:
            fc_sum_list[i] += fc_sum
            fc_square_list[i] += fc_square

def cal_save_uvb(fc_square_list, fc_sum_list, fc_count_list, fc_ratio, save_name = "fc1", save_path= "test"):
    for i in range(len(fc_square_list)):
        fc_square = fc_square_list[i]
        fc_sum = fc_sum_list[i]
        if fc_sum is not None:
            avg_tensor_i = fc_sum.reshape(-1, 1) / fc_count_list[i]

            cov_tensor_i = fc_square / fc_count_list[i] -  avg_tensor_i @ avg_tensor_i.T
            u,v, b  = cal_cov(cov_tensor_i, avg_tensor_i, fc_ratio[i])
            u_name = save_path + save_name  + "_"  + str(i) + "_u.pt"
            torch.save(u, u_name)
            v_name = save_path + save_name  + "_" + str(i) + "_v.pt"
            torch.save(v, v_name)
            b_name = save_path + save_name  + "_" + str(i) + "_avg.pt"
            torch.save(b, b_name)

@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    module_name = []
    hooks, hook_handles = [], []

    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hook = output_hook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))
            module_name.append(n)

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()


    fc1_square_list, fc1_sum_list, fc1_count_list = [None] * 36,  [None] * 36, [0] * 36
    fc2_square_list, fc2_sum_list,  fc2_count_list= [None] * 36, [None] * 36, [0] * 36
    qkv_square_list, qkv_sum_list,  qkv_count_list= [None] * 36, [None] * 36, [0] * 36
    fc1_ratio,fc2_ratio, qkv_ratio =  [0] * 36, [0] * 36, [0] * 36



    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        for hook in hooks:
            hook.clear()

        fc1_output_list = []
        fc2_output_list = []
        qkv_output_list = []

        output = model(images)
        output = output[0]
        for cal in range(len(module_name)):
            if "attn.qkv" in module_name[cal]:
                qkv_output_list.append(hooks[cal].outputs)
            if "mlp.fc1" in module_name[cal]:
                fc1_output_list.append(hooks[cal].outputs)
            if "mlp.fc2" in module_name[cal]:
                fc2_output_list.append(hooks[cal].outputs)

        cal_output_sum_square(fc1_output_list, fc1_square_list, fc1_sum_list, fc1_count_list)
        cal_output_sum_square(fc2_output_list, fc2_square_list, fc2_sum_list, fc2_count_list)
        cal_output_sum_square(qkv_output_list, qkv_square_list, qkv_sum_list, qkv_count_list)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

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
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    save_floader = "deit/ffn_output/all_fc_ada/deit_t/"
    if not os.path.exists(save_floader):
        os.makedirs(save_floader)
    save_path = save_floader + "deit_t_fewshot2_"
    cal_save_uvb(fc1_square_list, fc1_sum_list, fc1_count_list, fc1_ratio, "fc1",save_path )
    cal_save_uvb(fc2_square_list, fc2_sum_list, fc2_count_list, fc2_ratio, "fc2",save_path )
    cal_save_uvb(qkv_square_list, qkv_sum_list, qkv_count_list, qkv_ratio, "qkv",save_path )

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == '__main__':
    _, config = parse_option()

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

    main(config)
