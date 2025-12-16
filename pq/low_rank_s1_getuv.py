# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from timm.utils import AverageMeter

import sys
sys.path.append("/mnt/petrelfs/shaojie/code/DiT/")
from glob import glob
try:
    from apex import amp
except ImportError:
    amp = None


from distributed import init_distributed_mode
from pq.utils_model import parse_option, output_hook
from pq.low_rank_compress import cal_output_sum_square, cal_save_uvb
from pq.utils_model import init_data_and_model, init_env


def main(args):
    init_distributed_mode(args)
    rank, device, logger, experiment_dir = init_env(args)

    loader, model, state_dict, diffusion, vae = init_data_and_model(args, rank, device, logger)
    model = DDP(model.to(device), device_ids=[rank])
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    loss = getuv(args, loader, model, vae, diffusion, device, logger, experiment_dir)
    logger.info(f"Loss on dataset: {loss:.4f}")


@torch.no_grad()
def getuv(args, data_loader, model, vae, diffusion, device, logger, work_dir):

    model.eval()
    hooks, hook_handles, module_name = [], [], []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hook = output_hook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))
            module_name.append(n)

    batch_time, loss_meter = AverageMeter(), AverageMeter()
    end = time.time()

    num_layers = 28
    layer_dicts = {
        "fc1": {"square": [None] * num_layers, "sum": [None] * num_layers, "count": [0] * num_layers, "ratio": [0] * num_layers},
        "fc2": {"square": [None] * num_layers, "sum": [None] * num_layers, "count": [0] * num_layers, "ratio": [0] * num_layers},
        "qkv": {"square": [None] * num_layers, "sum": [None] * num_layers, "count": [0] * num_layers, "ratio": [0] * num_layers},
        "proj": {"square": [None] * num_layers, "sum": [None] * num_layers, "count": [0] * num_layers, "ratio": [0] * num_layers},
        "adaln": {"square": [None] * num_layers, "sum": [None] * num_layers, "count": [0] * num_layers, "ratio": [0] * num_layers},
    }

    iters = 1000
    for idx, (x, y) in enumerate(data_loader):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        for hook in hooks:
            hook.clear()
        with torch.no_grad():
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        model_kwargs = dict(y=y)
        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()

        output_lists = {"fc1": [], "fc2": [], "qkv": [], "proj": [], "adaln": []}

        for cal, name in enumerate(module_name):
            if "blocks" in name:
                for key in output_lists.keys():
                    if key in name:
                        output_lists[key].append(hooks[cal].outputs)

        for key, outputs in output_lists.items():
            cal_output_sum_square(outputs, layer_dicts[key]["square"], layer_dicts[key]["sum"], layer_dicts[key]["count"])

        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss_meter.update(loss.item() / dist.get_world_size(), y.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.log_every == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')

        if idx >= iters:
            break

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    save_path = os.path.join(work_dir, "dit_t_in_")
    for key in layer_dicts.keys():
        cal_save_uvb(layer_dicts[key]["square"], layer_dicts[key]["sum"], layer_dicts[key]["count"], layer_dicts[key]["ratio"], key, save_path, logger)

    return loss_meter.avg


if __name__ == '__main__':
    args = parse_option()
    main(args)
