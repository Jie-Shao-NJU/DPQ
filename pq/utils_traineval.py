import os
import math
import random
import torch
import json
import pickle
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import torch.distributed as dist
from PIL import Image
from time import time
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusion.gaussian_diffusion import _extract_into_tensor
import diffusion.gaussian_diffusion as gd
from diffusion import create_diffusion
from diffusion.respace import space_timesteps

from pq.utils_model import init_data, cleanup


class SideNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SideNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.fc2(out)
        return out

class CompensationBlock(nn.Module):
    def __init__(self, block, side_net, device):
        super(CompensationBlock, self).__init__()
        self.block = block
        self.side_net = side_net.to(device)

    def forward(self, x, c):
        out = self.block(x, c)
        side_out = self.side_net(x)
        out = out + side_out
        return out

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def sample(args, model_pq, vae, diffusion, sample_folder_dir):
    seed = dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    n = args.global_batch_size // dist.get_world_size()
    # n = args.global_batch_size // torch.cuda.device_count()
    global_batch_size = args.global_batch_size
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    latent_size = args.image_size // 8
    using_cfg = args.cfg_scale > 1.0
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")

    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    model_pq.to(device)
    model_pq.eval()

    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model_pq.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model_pq.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model_pq.forward

        # Sample images:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


class dit_generator:
    def __init__(self, timestep_respacing, latent_size, device):
        # create_diffusion
        betas = gd.get_named_beta_schedule('linear', 1000)
        use_timesteps=space_timesteps(1000, timestep_respacing)

        # SpacedDiffusion
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(betas)
        last_alpha_cumprod = 1.0
        new_betas = []
        self.set_alpha(betas)
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        self.set_alpha(new_betas)
        betas = np.array(new_betas, dtype=np.float64)

        # GaussianDiffusion
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        self.num_timesteps = int(betas.shape[0])
        self.latent_size = latent_size
        self.device = device

    def set_alpha(self, betas):
        # GaussianDiffusion
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

    def forward_val(self, vae, models, cfg=False, name="sample_pq", args=None, logger=None):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # class_labels = [(i+200) for i in range(8)]
        class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
        z, model_kwargs = self.pre_process(class_labels, cfg=cfg, args=args)

        imgs = [z.clone() for _ in models]
        indices = list(range(self.num_timesteps))[::-1]
        indices_tqdm = tqdm(indices)
        mse_losses = [[] for _ in models[1:]]
        block_errors_all_models = [{block_id: [] for block_id in range(28)} for _ in models[1:]]

        for i in indices_tqdm:
            t = torch.tensor([i] * z.shape[0], device=self.device)
            with torch.no_grad():
                map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
                new_ts = map_tensor[t]
                model_outputs = []
                features = []
                for model, img in zip(models, imgs):
                    model_output, feat = model(img, new_ts, **model_kwargs, distill=True, stat=False)
                    model_outputs.append(model_output)
                    features.append(feat)
                imgs = self.post_process(t, imgs, model_outputs)

                for model_id in range(1, len(models)):
                    for block_id, (output, output_ref) in enumerate(zip(features[model_id], features[0])):
                        block_error = torch.mean((output - output_ref) ** 2).cpu().numpy()
                        block_errors_all_models[model_id - 1][block_id].append(block_error)
                for model_id in range(1, len(models)):
                    mse_loss = torch.mean((model_outputs[model_id] - model_outputs[0]) ** 2).cpu().numpy()
                    mse_losses[model_id - 1].append(mse_loss)

        samples = [vae.decode(img / 0.18215).sample for img in imgs]
        for idx, sample in enumerate(samples):
            # sample, _ = sample.chunk(2, dim=0)
            save_image(sample, f"{name}/model_{idx}.png", nrow=4, normalize=True, value_range=(-1, 1))
            logger.info(f"Model {idx} sample saved as {name}/model_{idx}.png")

        with open(f"{name}/block_errors.pkl", "wb") as pkl_file:
            pickle.dump(block_errors_all_models, pkl_file)
        with open(f"{name}/indices.pkl", "wb") as pkl_file:
            pickle.dump(indices, pkl_file)
            
        for block_id in range(28):
            plt.figure(figsize=(10, 6))
            for model_id, block_errors in enumerate(block_errors_all_models, start=1):
                errors = block_errors[block_id]
                plt.plot(indices[::-1], errors, label=f"Model {model_id} vs. Model 0")
            plt.xlabel("Step")
            plt.ylabel("MSE Error")
            plt.legend(loc='upper left', fontsize='small')
            plt.title(f"MSE Error Trend for Block {block_id} (Comparison Across Models)")
            if not os.path.exists(f"{name}/perblock/"):
                os.makedirs(f"{name}/perblock/")
            plt.savefig(f"{name}/perblock/block_{block_id}_mse.png", bbox_inches='tight')
            plt.close()
        return samples

    def pre_process(self, class_labels, cfg=False, args=None):
        n = len(class_labels)
        z = torch.randn(n, 4, self.latent_size, self.latent_size, device=self.device)
        y = torch.tensor(class_labels, device=self.device)
        if cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=self.device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        else:
            model_kwargs = dict(y=y)
        return z, model_kwargs

    def post_process(self, t, imgs, model_outputs):
        B, C = imgs[0].shape[:2]
        samples = []
        min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, imgs[0].shape)
        max_log = _extract_into_tensor(np.log(self.betas), t, imgs[0].shape)
        noise = torch.randn_like(imgs[0])
        for img, model_output in zip(imgs, model_outputs):
            model_output_split, model_var_values = torch.split(model_output, C, dim=1)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            pred_xstart = self._predict_xstart_from_eps(x_t=img, t=t, eps=model_output_split)
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=img, t=t)
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))  # no noise when t == 0
            sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
            samples.append(sample)
        return samples

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    

def train(args, logger, model, vae, diffusion, checkpoint_dir):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    loader, sampler = init_data(args, rank, logger)

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            # loss_dict = diffusion.training_losses_distill(model_pq, model, x, t, model_kwargs)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    if not os.path.exists(checkpoint_dir):
                        os.mkdir(checkpoint_dir)
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    logger.info("Done!")


def train_qwerty(args, logger, model_origin, model, vae, checkpoint_dir):
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    loader, sampler = init_data(args, rank, logger)
    latent_size = args.image_size // 8
    diffusion_qwerty = create_diffusion(timestep_respacing="")
    diffusion_gen = dit_generator('250', latent_size=latent_size, device=device)

    for block_id in range(len(model.blocks)):
        side_net = SideNetwork(1152, 4*1152, 1152)
        model.blocks[block_id] = CompensationBlock(block=model.blocks[block_id], side_net=deepcopy(side_net),  device=device)

    model.eval()
    for block in model.blocks:
        for param in block.parameters():
            param.requires_grad = False

    for block in model.blocks:
        block.side_net.parameters()
        for param in block.side_net.parameters():
            param.requires_grad = True
    opt = torch.optim.AdamW(
        [param for block in model.blocks for param in block.side_net.parameters() if param.requires_grad],
        lr=1e-4,
        weight_decay=0
    )
    model = DDP(model.to(device), device_ids=[rank])

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion_qwerty.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            # loss_dict = diffusion.training_losses_distill(model_pq, model, x, t, model_kwargs)
            loss_dict = diffusion_qwerty.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    if not os.path.exists(checkpoint_dir):
                        os.mkdir(checkpoint_dir)
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

        if rank == 0:
            image_name = f"pq_qwerty_epoch{epoch}"
            image_dir = f"{os.path.dirname(checkpoint_dir)}/gen"
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            diffusion_gen.forward_val(vae, model_origin.forward, model.forward, cfg=False, name=f"{image_dir}/{image_name}", logger=logger)

    model.eval()  # important! This disables randomized embedding dropout
    logger.info("Done!")


def opt_pq(model_ori, model_comp, logger, args, device):
    fc_params = {
        1: ("fc1", "mlp", "fc1"),
        2: ("fc2", "mlp", "fc2"),
        3: ("qkv", "attn", "qkv"),
        4: ("proj", "attn", "proj"),
        5: ("adaln", "adaLN_modulation", "1")
    }
    for block_id in range(len(model_comp.blocks)):
        block_comp = model_comp.blocks[block_id]
        block_ori = model_ori.blocks[block_id]
        for fc_idx in range(1, 6):
            file_name, layer_name, org_name = fc_params[fc_idx]
            weight_org = deepcopy(getattr(getattr(block_ori, layer_name), org_name).weight).detach().to(device)
            weight_opt = deepcopy(getattr(getattr(block_comp, layer_name), org_name)).to(device)

            weight_comp = weight_opt._get_uncompressed_weight()
            error = (weight_comp - weight_org).abs().mean().item()
            ori_abs = weight_org.abs().mean().item()
            logger.info(f"[Block({block_id})-Layer({file_name})] Ori abs: {ori_abs:.4f}, L1 Error: {error:.4f}")

            weight_opt.bias.requires_grad = False
            opt = torch.optim.AdamW(weight_opt.parameters(), lr=1e-2, weight_decay=0.02)
            # criterion = nn.MSELoss(reduction='sum')
            for epoch in range(args.epochs):
                loss = weight_opt.get_loss(weight_org)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if epoch % 10 == 0:
                    logger.info(f"[Block({block_id})-Layer({file_name})] MSE Loss: {loss.item():.4f}")
            
            if fc_idx == 1:
                model_comp.blocks[block_id].mlp.fc1 = deepcopy(weight_opt).cuda()
            if fc_idx == 2:
                model_comp.blocks[block_id].mlp.fc2 = deepcopy(weight_opt).cuda()
            if fc_idx == 3:
                model_comp.blocks[block_id].attn.qkv = deepcopy(weight_opt).cuda()
            if fc_idx == 4:
                model_comp.blocks[block_id].attn.proj = deepcopy(weight_opt).cuda()
            if fc_idx == 5:
                model_comp.blocks[block_id].adaLN_modulation[1] = deepcopy(weight_opt).cuda()

            weight_comp = weight_opt._get_uncompressed_weight()
            error = (weight_comp - weight_org).abs().mean().item()
            ori_abs = weight_org.abs().mean().item()
            logger.info(f"[Block({block_id})-Layer({file_name})] Ori abs: {ori_abs:.4f}, L1 Error: {error:.4f}")
    return


def save_ckpt(model, args, checkpoint_dir, logger):
    checkpoint = {
        "model": model.state_dict(),
        "args": args
    }
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/ckpt.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")