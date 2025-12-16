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
import torch.nn as nn
from torch import optim as optim

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_fewshot_loader, build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, DSDTraining

from models.swin_transformer_pca_adaptive import SwinTransformer as SwinTransformerAda
from models.swin_transformer import SwinTransformer

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
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
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
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_fewshot_loader(config)
    # dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    # model = build_model(config)

    # qkv_len = [[32, 72], [144, 144], [160, 160, 160, 160, 288, 192, 288, 192, 288, 288, 288, 288, 288, 288, 288, 160, 192, 160], [192, 256]]
    # fc1_len = [[64, 76.8], [153.6, 153.6], [192, 224, 224, 224, 224, 307.2, 307.2, 192, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [320, 96]]
    # fc2_len = [[76.8, 76.8], [153.6, 153.6], [224, 224, 192, 160, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [480, 448]]
    # qkv_use_uv = [[True, False], [False, False], [True, True, True, True, False, True, False, True, False, False, False, False, False, False, False, True, True, True], [True, True]]
    # fc1_use_uv = [[True, False], [False, False], [True, True, True, True, True, False, False, True, False, False, False, False, False, False, False, False, False, False], [True, True]]
    # fc2_use_uv = [[False, False], [False, False], [True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [True, True]]

    # qkv_len = [[32, 72], [144, 144], [160, 160, 160, 160, 288, 288, 288, 192, 288, 288, 288, 288, 288, 288, 288, 288, 288, 160], [192, 256]]
    # fc1_len = [[76.8, 76.8], [153.6, 153.6], [192, 224, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [352, 96]]
    # fc2_len = [[76.8, 76.8], [153.6, 153.6], [307.2, 307.2, 224, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [614.4, 448]]
    # qkv_use_uv = [[True, False], [False, False], [True, True, True, True, False, False, False, True, False, False, False, False, False, False, False, False, False, True], [True, True]]
    # fc1_use_uv = [[False, False], [False, False], [True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [True, True]]
    # fc2_use_uv = [[False, False], [False, False], [False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, True]]


    # qkv_len = [[32, 72], [144, 144], [160, 160, 160, 160, 288, 160, 288, 160, 288, 192, 288, 160, 288, 288, 288, 128, 160, 160], [160, 224]]
    # fc1_len = [[64, 76.8], [153.6, 153.6], [192, 192, 192, 192, 224, 224, 192, 192, 224, 224, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [320, 96]]
    # fc2_len = [[76.8, 76.8], [153.6, 153.6], [224, 192, 160, 128, 192, 192, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [448, 416]]
    # qkv_use_uv = [[True, False], [False, False], [True, True, True, True, False, True, False, True, False, True, False, True, False, False, False, True, True, True], [True, True]]
    # fc1_use_uv = [[True, False], [False, False], [True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False], [True, True]]
    # fc2_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False], [True, True]]


    # qkv_len =[[32, 72], [144, 144], [160, 160, 128, 160, 128, 128, 192, 160, 160, 128, 160, 128, 160, 160, 160, 128, 160, 128], [160, 224]]
    # fc1_len =[[64, 76.8], [153.6, 153.6], [192, 192, 192, 192, 224, 192, 160, 192, 160, 192, 224, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [288, 96]]
    # fc2_len = [[76.8, 76.8], [153.6, 153.6], [224, 192, 160, 128, 160, 160, 128, 160, 160, 192, 192, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2, 307.2], [416, 384]]
    # qkv_use_uv =[[True, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc1_use_uv =[[True, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False], [True, True]]
    # fc2_use_uv =[[False, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False], [True, True]]

    # proj_len = [[64,64],[128,128],[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256],[512,512]]
    # proj_use_uv = [[False, False], [False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False]]

    # qkv_len =  [[48, 48], [96, 96], [192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192], [384, 384]]
    # fc1_len = [[48, 48], [96, 96], [192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192], [384, 384]]
    # fc2_len = [[48, 48], [96, 96], [192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192], [384, 384]]
    # proj_len = [[48, 48], [96, 96], [192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192], [384, 384]]
    # qkv_len =  [[64, 64], [128, 128], [224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224], [416, 416]]
    # fc1_len = [[64, 64], [128, 128], [224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224], [416, 416]]
    # fc2_len = [[64, 64], [128, 128], [224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224], [416, 416]]
    # proj_len = [[64, 64], [128, 128], [224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224], [416, 416]]

    # qkv_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc1_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc2_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # proj_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]

    # model = SwinTransformerAda(embed_dim= 96, depths= [ 2, 2, 18, 2 ],num_heads = [ 3, 6, 12, 24 ],window_size= 7,drop_path_rate=config.MODEL.DROP_PATH_RATE, 
    #         qkv_use_uv = qkv_use_uv, proj_use_uv=proj_use_uv,fc1_use_uv = fc1_use_uv,fc2_use_uv = fc2_use_uv,
    #         qkv_len = qkv_len, proj_len=proj_len, fc1_len = fc1_len, fc2_len = fc2_len)
    # teacher_model = SwinTransformer(embed_dim= 96, depths= [ 2, 2, 18, 2 ],num_heads = [ 3, 6, 12, 24 ],window_size= 7,drop_path_rate=0)


    # qkv_len =  [[96, 96], [192, 192], [192, 192, 192, 160, 192, 160, 384, 192, 224, 224, 256, 224, 192, 224, 256, 224, 192, 128], [32, 64]]
    # fc1_len = [[102.4, 102.4], [204.8, 204.8], [224, 256, 256, 288, 224, 256, 256, 256, 409.6, 288, 320, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 288], [288, 544]]
    # fc2_len = [[102.4, 102.4], [204.8, 204.8], [352, 256, 288, 256, 288, 256, 256, 288, 409.6, 288, 224, 224, 224, 256, 409.6, 409.6, 409.6, 409.6], [704, 672]]
    # qkv_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc1_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, True, True, False, True, True, False, False, False, False, False, False, True], [True, True]]
    # fc2_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, True, True, False, True, True, True, True, True, False, False, False, False], [True, True]]

    qkv_len =  [[64, 96], [160, 192], [192, 192, 192, 160, 256, 192, 256, 192, 224, 224, 224, 256, 256, 256, 256, 224, 224, 160], [192, 256]]
    fc1_len = [[768, 768], [768, 768], [224, 224, 224, 256, 256, 256, 224, 224, 256, 224, 288, 288, 768, 768, 768, 768, 768, 320], [352, 544]]
    fc2_len = [[768, 768], [768, 768], [256, 224, 224, 224, 224, 224, 288, 288, 288, 256, 288, 320, 320, 320, 768, 768, 768, 768], [544, 480]]
    qkv_use_uv = [[True, False], [True, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    fc1_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, True], [True, True]]
    fc2_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False], [True, True]]

    # qkv_len = [[64, 96], [160, 192], [192, 192, 224, 192, 256, 256, 288, 224, 288, 256, 384, 256, 384, 384, 384, 256, 256, 192], [224, 288]]
    # fc1_len = [[96, 102.4], [204.8, 204.8], [224, 224, 288, 288, 288, 288, 288, 288, 288, 288, 320, 352, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6], [384, 576]]
    # fc2_len = [[102.4, 102.4], [204.8, 204.8], [288, 256, 224, 256, 320, 288, 320, 352, 320, 320, 320, 320, 320, 409.6, 409.6, 409.6, 409.6, 409.6], [576, 544]]
    # qkv_use_uv = [[True, False], [True, False], [True, True, True, True, True, True, True, True, True, True, False, True, False, False, False, True, True, True], [True, True]]
    # fc1_use_uv = [[True, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False], [True, True]]
    # fc2_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False], [True, True]]

    # qkv_len = [[64, 96], [160, 192], [192, 224, 256, 288, 288, 256, 384, 256, 288, 288, 384, 384, 384, 384, 384, 288, 288, 224], [224, 288]]
    # fc1_len = [[96, 102.4], [204.8, 204.8], [256, 256, 288, 288, 288, 320, 320, 320, 320, 320, 320, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6], [416, 576]]
    # fc2_len = [[102.4, 102.4], [204.8, 204.8], [288, 288, 256, 288, 320, 320, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6], [640, 576]]
    # qkv_use_uv = [[True, False], [True, False], [True, True, True, True, True, True, False, True, True, True, False, False, False, False, False, True, True, True], [True, True]]
    # fc1_use_uv = [[True, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False], [True, True]]
    # fc2_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False], [True, True]]


    proj_len = [[64,64],[128,128],[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256],[512,512]]
    proj_use_uv = [[False, False], [False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False]]


    # qkv_len =  [[64, 64], [128, 128], [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256], [512, 512]]
    # fc1_len = [[64, 64], [128, 128], [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256], [512, 512]]
    # fc2_len = [[64, 64], [128, 128], [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256], [512, 512]]
    # qkv_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc1_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc2_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # proj_len = [[64,64],[128,128],[256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256],[512,512]]
    # proj_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]

    # qkv_len = [[96, 32], [192, 192], [384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384], [32, 32]]
    # fc1_len = [[64, 102.4], [204.8, 204.8], [409.6, 409.6, 409.6, 288, 409.6, 384, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 32, 64, 409.6], [32, 32]]
    # fc2_len = [[102.4, 102.4], [96, 204.8], [409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 32, 32, 32, 32, 409.6], [32, 32]]
    # qkv_use_uv = [[False, True], [False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False]]
    # fc1_use_uv = [[True, False], [False, False], [False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False], [False, False]]
    # fc2_use_uv = [[False, False], [True, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False]]

    # qkv_len = [[96, 96], [192, 192], [384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384], [32, 32]]
    # fc1_len = [[102.4, 102.4], [204.8, 204.8], [409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 32, 64, 409.6], [32, 32]]
    # fc2_len = [[102.4, 102.4], [204.8, 204.8], [409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 409.6, 32, 32, 32, 32, 409.6], [32, 32]]
    # qkv_use_uv = [[False, False], [False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [True, True]]
    # fc1_use_uv = [[False, False], [False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False], [True, True]]
    # fc2_use_uv = [[False, False], [False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, False], [True, True]]
    model = SwinTransformerAda(embed_dim= 128, depths= [ 2, 2, 18, 2 ],num_heads = [ 4, 8, 16, 32 ],window_size= 7, drop_path_rate=config.MODEL.DROP_PATH_RATE, 
            qkv_use_uv = qkv_use_uv, proj_use_uv=proj_use_uv,fc1_use_uv = fc1_use_uv,fc2_use_uv = fc2_use_uv,
            qkv_len = qkv_len, proj_len=proj_len, fc1_len = fc1_len, fc2_len = fc2_len)
    teacher_model = SwinTransformer(embed_dim= 128, depths= [ 2, 2, 18, 2 ],num_heads = [ 4, 8, 16, 32 ],window_size= 7,drop_path_rate=0)
    # model = SwinTransformer(embed_dim= 128, depths= [ 2, 2, 18, 2 ],num_heads = [ 4, 8, 16, 32 ],window_size= 7, drop_path_rate=config.MODEL.DROP_PATH_RATE)




    # qkv_len = [[64, 144], [192, 288], [160, 192, 256, 224, 288, 256, 288, 288, 352, 352, 384, 320, 384, 320, 384, 320, 576, 320], [480, 608]]
    # fc1_len = [[96, 153.6], [288, 307.2], [224, 256, 256, 288, 320, 352, 384, 352, 352, 384, 384, 352, 384, 448, 480, 480, 614.4, 614.4], [704, 1228.8]]
    # fc2_len = [[153.6, 153.6], [307.2, 256], [256, 224, 192, 256, 256, 256, 320, 320, 320, 352, 352, 384, 416, 448, 480, 416, 480, 448], [1228.8, 992]]
    # qkv_use_uv = [[True, False], [True, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True], [True, True]]
    # fc1_use_uv = [[True, False], [True, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False], [True, False]]
    # fc2_use_uv = [[False, False], [False, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [False, True]]


    # qkv_len = [[64, 144], [192, 288], [160, 256, 256, 256, 288, 288, 288, 320, 384, 352, 384, 352, 416, 320, 576, 416, 576, 416], [544, 672]]
    # fc1_len = [[96, 153.6], [307.2, 307.2], [224, 288, 256, 320, 320, 384, 416, 352, 416, 384, 448, 448, 448, 480, 614.4, 614.4, 614.4, 614.4], [800, 1228.8]]
    # fc2_len = [[153.6, 153.6], [307.2, 307.2], [256, 288, 256, 288, 288, 352, 352, 384, 416, 384, 416, 480, 480, 614.4, 614.4, 614.4, 614.4, 480], [1228.8, 1228.8]]
    # qkv_use_uv = [[True, False], [True, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, False, True], [True, True]]
    # fc1_use_uv = [[True, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False], [True, False]]
    # fc2_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, True], [False, False]]

    # qkv_len = [[64, 144], [224, 288], [160, 256, 352, 288, 352, 352, 352, 384, 416, 384, 576, 384, 576, 416, 576, 576, 576, 576], [576, 736]]
    # fc1_len = [[96, 153.6], [307.2, 307.2], [256, 288, 288, 352, 352, 416, 416, 448, 448, 480, 448, 480, 512, 614.4, 614.4, 614.4, 614.4, 614.4], [800, 1228.8]]
    # fc2_len = [[153.6, 153.6], [307.2, 307.2], [256, 320, 320, 320, 320, 384, 384, 416, 448, 416, 480, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4, 614.4], [1228.8, 1228.8]]
    # qkv_use_uv = [[True, False], [True, False], [True, True, True, True, True, True, True, True, True, True, False, True, False, True, False, False, False, False], [True, True]]
    # fc1_use_uv = [[True, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False], [True, False]]
    # fc2_use_uv = [[False, False], [False, False], [True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False], [False, False]]



    # qkv_len =  [[96, 96], [192, 192], [384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384], [768, 768]]
    # fc1_len = [[96, 96], [192, 192], [384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384], [768, 768]]
    # fc2_len = [[96, 96], [192, 192], [384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384], [768, 768]]
    # qkv_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc1_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # fc2_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]
    # proj_len = [[96, 96], [192, 192], [384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384], [768, 768]]
    # proj_use_uv = [[True, True], [True, True], [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [True, True]]



    # model = SwinTransformerAda(embed_dim= 192, depths= [ 2, 2, 18, 2 ],num_heads = [ 6, 12, 24, 48 ],window_size= 7, drop_path_rate=config.MODEL.DROP_PATH_RATE, qkv_use_uv = qkv_use_uv, proj_use_uv=proj_use_uv,fc1_use_uv = fc1_use_uv,fc2_use_uv = fc2_use_uv,
    #         qkv_len = qkv_len, proj_len=proj_len, fc1_len = fc1_len, fc2_len = fc2_len)
    # teacher_model = SwinTransformer(embed_dim= 192, depths= [ 2, 2, 18, 2 ],num_heads = [ 6, 12, 24, 48 ],window_size= 7,drop_path_rate=0)



    model.cuda()
    teacher_model.cuda()
    logger.info(str(model))

    # optimizer = build_optimizer(config, model)

    # optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
    #                                 lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    for param in model.named_parameters():
        if "head.weight" in param[0] or "head.bias" in param[0]:
            print(param[0])
            param[1].requires_grad = False
        else:
            param[1].requires_grad = True

    for param in model.named_parameters():
        print(param[0])
        if not param[1].requires_grad:
            print(param[0])

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

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
