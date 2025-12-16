import time
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import torch
import torch.distributed as dist
from timm.utils import AverageMeter

import sys
sys.path.append("/mnt/petrelfs/shaojie/code/DiT/")
from models import DiT_models
from pq.low_rank_models import DiT_uv_models
from pq.low_rank_compress import reset_param, merge_model
from pq.utils_model import parse_option, init_env, init_data_and_model
from pq.utils_traineval import dit_generator

from distributed import init_distributed_mode
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from apex import amp
except ImportError:
    amp = None


def main(args):
    init_distributed_mode(args)
    rank, device, logger, experiment_dir = init_env(args)

    loader, model, state_dict, diffusion, vae = init_data_and_model(args, rank, device, logger)
    latent_size = args.image_size // 8
    load_path = "results/low_rank/002-DiT-XL-2/dit_t_in_"
    mode = args.s2_mode
    if mode == "mse":
        evaluator = dit_evaluator('250', latent_size, device)
        model_t = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes
        ).to(device)
        msg = model_t.load_state_dict(state_dict)
        model_t = DDP(model_t, device_ids=[rank])
        model_t.eval()


    num_layers = 28
    dims_dict = {
        1: [921.6] + np.arange(896, 32, -32).tolist(),
        2: [921.6] + np.arange(896, 32, -32).tolist(),
        3: [864] + np.arange(832, 32, -32).tolist(),
        4: [576] + np.arange(544, 32, -32).tolist(),
        5: [987.4] + np.arange(960, 32, -32).tolist()
    }

    for fc_i in range(1, 6):
        for block_i in range(28):

            fc1_len, fc2_len, qkv_len, proj_len, adaln_len, fc1_use_uv, fc2_use_uv, qkv_use_uv, proj_use_uv, adaln_use_uv = reset_param(num_layers)
            dims = dims_dict[fc_i]

            use_uv_dict = {
                1: fc1_use_uv,
                2: fc2_use_uv,
                3: qkv_use_uv,
                4: proj_use_uv,
                5: adaln_use_uv
            }

            use_uv_dict[fc_i][block_i] = True

            for dim_i, dim in enumerate(dims):
                len_dict = {
                    1: fc1_len,
                    2: fc2_len,
                    3: qkv_len,
                    4: proj_len,
                    5: adaln_len
                }
                len_dict[fc_i][block_i] = dim

                if dim_i == 0:
                    model = DiT_models[args.model](
                        input_size=latent_size,
                        num_classes=args.num_classes
                    ).to(device)
                    msg = model.load_state_dict(state_dict)
                else:
                    model = DiT_uv_models[args.model](
                        input_size=latent_size,
                        num_classes=args.num_classes,
                        qkv_use_uv=qkv_use_uv, proj_use_uv=proj_use_uv,
                        fc1_use_uv=fc1_use_uv, fc2_use_uv=fc2_use_uv, adaln_use_uv=adaln_use_uv,
                        qkv_len=qkv_len, proj_len=proj_len, fc1_len=fc1_len, fc2_len=fc2_len, adaln_len=adaln_len
                    ).to(device)

                    state_dict_merge = merge_model(deepcopy(state_dict), block_i, fc_i, dim, load_path, logger)
                    msg = model.load_state_dict(state_dict_merge)

                n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"number of params: {n_parameters}")
                logger.info(msg)
                model = DDP(model, device_ids=[rank])

                if mode == "diffu":
                    loss = get_score_diffuloss(args, loader, model, vae, diffusion, device, logger, experiment_dir)
                elif mode == "mse":
                    loss = evaluator.forward_val(model_t.forward, model.forward)
                with open(f"{experiment_dir}/dim_loss/{block_i}_{fc_i}.txt", 'a') as file:
                    file.write(f"{dim}, {loss}\n")
                if block_i == 27 and fc_i == 5:
                    logger.info("Done.")
                else:
                    del model
    
@torch.no_grad()
def get_score_diffuloss(args, data_loader, model, vae, diffusion, device, logger, work_dir):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    end = time.time()

    for idx, (x, y) in enumerate(data_loader):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        with torch.no_grad():
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        # t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        t = torch.linspace(0, diffusion.num_timesteps - 1, steps=x.shape[0], device=device).long()
        model_kwargs = dict(y=y)

        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss = loss.item() / dist.get_world_size()

        loss_meter.update(loss, y.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    logger.info(
        f'Test: [{idx}/{len(data_loader)}]\t'
        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
        f'Mem {memory_used:.0f}MB')
    
    return loss_meter.avg


class dit_evaluator(dit_generator):
    def __init__(self, timestep_respacing, latent_size, device):
        super(dit_evaluator, self).__init__(timestep_respacing, latent_size, device)

    def forward_val(self, model, model_uv, cfg=False):
        class_labels = list(range(16))
        bs = 8
        mse_loss_meter = AverageMeter()

        for i in range(0, len(class_labels), bs):
            label = class_labels[i:i + bs]
            z, model_kwargs = self.pre_process(label, cfg=cfg)
            img = z
            img_pq = z
            indices = list(range(self.num_timesteps))[::-1]
            indices_tqdm = tqdm(indices)

            for i in indices_tqdm:
                t = torch.tensor([i] * z.shape[0], device=self.device)
                with torch.no_grad():
                    map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
                    new_ts = map_tensor[t]
                    model_output = model(img, new_ts, **model_kwargs)
                    model_output_pq = model_uv(img_pq, new_ts, **model_kwargs)
                    img, img_pq = self.post_process(t, img, img_pq, model_output, model_output_pq)

            mse_loss = torch.mean((img - img_pq) ** 2)
            mse_loss_meter.update(mse_loss.item(), 1)

        return mse_loss_meter.avg
    

if __name__ == '__main__':
    args = parse_option()
    main(args)