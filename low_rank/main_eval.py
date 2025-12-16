# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import random
import argparse
import datetime
import numpy as np

import torch
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
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

# from models.swin_transformer_svd_enlarge import SwinTransformer
# from models.swin_transformer_pca_merge import SwinTransformer
# from models.swin_transformer_pca import SwinTransformer
# from models.swin_transformer_pca_adaptive import SwinTransformer
from models.swin_transformer import SwinTransformer
# from models.swin_transformer_pca_correct import SwinTransformer
# from models.swin_transformer_pca_correct import SwinTransformer as SwinTransformer_correct

# from models.resmlp_models_svd import resmlp_12
from models.resmlp_models import resmlp_12, resmlpB_24
# from models.resmlp_models_pca_correct import resmlp_12
# from models.resmlp_models_eval import resmlp_12
# from models.resmlp_models_pca_merge import resmlp_12, resmlp_36

# from models.levit_model import LeViT_128S
# from models.levit_model_wo_bn import LeViT_128 as LeViT_128_wobn
# from models.levit_model_wo_bn_pca import LeViT_128 as LeViT_128_wobn_pca
# from models.levit_model_wo_bn_pca import LeViT_384 as LeViT_384_wobn_pca
# from models.levit_model_wo_bn_pca import LeViT_256 as LeViT_256_wobn_pca

from models.deit_model import VisionTransformer
# from models.deit_model_pca_adaptive import VisionTransformer
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

    # easy config modification
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
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--sparse-rate', type=float, default=0.1)
    parser.add_argument('--sparse-freq', type=int, default=100)

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)


    # logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    # model = build_model(config)
    # model = resmlp_12(drop_path_rate=0, fc1_lens=[120,168,208,248,248,272,280,288,280,264,272,1536])
    # model = resmlp_36(drop_path_rate=0)

    # model = SwinTransformer(drop_path_rate=0)
    # model = SwinTransformer_correct(embed_dim= 128, depths= [ 2, 2, 18, 2 ],num_heads = [ 4, 8, 16, 32 ],window_size= 7)
    # model = SwinTransformer(embed_dim= 128, depths= [ 2, 2, 18, 2 ],num_heads = [ 4, 8, 16, 32 ],window_size= 7)
    model = SwinTransformer(embed_dim= 96, depths= [ 2, 2, 18, 2 ],num_heads = [ 3, 6, 12, 24 ],window_size= 7)
    # model = SwinTransformer(embed_dim= 96, depths= [ 2, 2, 6, 2 ],num_heads = [ 3, 6, 12, 24 ],window_size= 7)

    # qkv_len =  [[64, 64], [128, 128], [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256], [512, 512]]
    # fc1_len = [[64, 64], [128, 128], [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256], [512, 512]]
    # fc2_len = [[64, 64], [128, 128], [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256], [512, 512]]
    # qkv_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc1_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc2_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # proj_len = [[64,64],[128,128],[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256],[512,512]]
    # proj_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]

    # model = SwinTransformer(embed_dim= 128, depths= [ 2, 2, 18, 2 ],num_heads = [ 4, 8, 16, 32 ],window_size= 7, drop_path_rate=0,
    #         qkv_use_uv = qkv_use_uv, proj_use_uv=proj_use_uv,fc1_use_uv = fc1_use_uv,fc2_use_uv = fc2_use_uv,
    #         qkv_len = qkv_len, proj_len=proj_len, fc1_len = fc1_len, fc2_len = fc2_len)

    # qkv_len =  [[96, 96], [192, 192], [384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384], [768, 768]]
    # fc1_len = [[96, 96], [192, 192], [384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384], [768, 768]]
    # fc2_len = [[96, 96], [192, 192], [384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384], [768, 768]]
    # qkv_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc1_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc2_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # proj_len = [[96, 96], [192, 192], [384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384], [768, 768]]
    # proj_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]


    # model = SwinTransformer(embed_dim= 192, depths= [ 2, 2, 18, 2 ],num_heads = [ 6, 12, 24, 48 ],window_size= 7,
            # qkv_use_uv = qkv_use_uv, proj_use_uv=proj_use_uv,fc1_use_uv = fc1_use_uv,fc2_use_uv = fc2_use_uv,
            # qkv_len = qkv_len, proj_len=proj_len, fc1_len = fc1_len, fc2_len = fc2_len)

    # model = LeViT_128S_wobn( distillation = True)
    # model = LeViT_256_wobn_pca( distillation = True)
    # fc1_len = [128,128, 128,128, 128,128, 128,128, 128,128, 128,128]
    # fc2_len = [128,128, 128,128, 128,128, 128,128, 128,128, 128,128]
    # qkv_len = [128,128, 128,128, 128,128, 128,128, 128,128, 128,128]
    # proj_len = [128,128, 128,128, 128,128, 128,128, 128,128, 128,128]
    # fc2_len =  [136, 128, 128, 128, 128, 128, 128, 128, 128, 64, 64, 64]

    # fc2_len =  [96,96, 96,96, 96,96, 96,96, 96,96, 96,96]

    # fc1_use_uv = [False, False, False, False, False, False, False, False,  False, False,  False, False, False]
    # fc2_use_uv = [True, True, True, True, True, True, True, True,  True, True,  True, True, True]
    # qkv_use_uv = [False, False, False, False, False, False, False, False,  False, False,  False, False, False]

    # qkv_len = [320 ] * 12
    # fc1_len = [320 ] * 12
    # fc2_len = [320 ] * 12
    # qkv_use_uv = [True] * 12
    # fc1_use_uv = [True] * 12
    # fc2_use_uv = [True]  * 12

    # model = VisionTransformer(patch_size=16,  embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
    #         qkv_use_uv = qkv_use_uv,fc1_use_uv = fc1_use_uv,fc2_use_uv = fc2_use_uv,qkv_len = qkv_len, fc1_len = fc1_len, fc2_len = fc2_len)
    # model = resmlpB_24()
    model = VisionTransformer(patch_size=16,  embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True)

    model.cuda()
    logger.info(str(model))

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")


    if config.MODEL.RESUME:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        if 'model' in checkpoint.keys():
            msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        else:
            msg = model_without_ddp.load_state_dict(checkpoint, strict=False)

        logger.info(msg)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()

    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # print(target)
        # new_s = images.transpose(0,1)
        # print("mean 0: ",torch.mean(new_s[0]))
        # print("mean 1: ",torch.mean(new_s[1]))
        # print("mean 2: ",torch.mean(new_s[2]))
        # print("var 0: ",torch.var(new_s[0]))
        # print("var 1: ",torch.var(new_s[1]))
        # print("var 2: ",torch.var(new_s[2]))

        # print(torch.var(images))
        # compute output
        outputs = model(images)
        output = outputs[0]
        # output = outputs
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
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


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

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    # if dist.get_rank() == 0:
    #     path = os.path.join(config.OUTPUT, "config.json")
    #     with open(path, "w") as f:
    #         f.write(config.dump())
    #     logger.info(f"Full config saved to {path}")

    # print config
    # logger.info(config.dump())

    main(config)
