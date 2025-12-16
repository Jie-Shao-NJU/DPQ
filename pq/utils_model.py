import torch
import random
import os
from glob import glob
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from diffusers.models import AutoencoderKL

from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from pqf.utils.model_size import compute_model_nbits
from pqf.utils.config_loader import load_config_clean
from pqf.compression.model_compression import compress_model


def create_logger(logging_dir):
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def cleanup():
    dist.destroy_process_group()


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


def plot_3d_surface(matrix, ax, title, vmin, vmax):
    """Efficient 3D surface plotting."""
    x = np.arange(matrix.shape[0])
    y = np.arange(matrix.shape[1])
    X, Y = np.meshgrid(x, y)
    Z = matrix.cpu().numpy()
    ax.plot_surface(X, Y, Z.T, cmap='viridis')

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)


def calculate_outliers(matrix, logger, name):
    matrix_np = matrix.cpu().numpy()
    mean = np.mean(matrix_np)
    std = np.std(matrix_np)
    
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    outliers = np.logical_or(matrix_np < lower_bound, matrix_np > upper_bound)
    outlier_ratio = np.sum(outliers) / matrix_np.size

    logger.info(f"{name} - Min: {matrix_np.min()}, Max: {matrix_np.max()}, Outliers: {outlier_ratio*100:.2f}%")
    return matrix_np.min(), matrix_np.max(), outlier_ratio


def vis_weights(model, logger, save_dir, mode='uv'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    prefix_layers = {}

    for name, param in model.named_parameters():
        if mode == 'uv':
            if '_u' in name or '_v' in name:
                prefix = name.split('_')[0]
                max_abs_value = param.abs().max().item()
                logger.info(f"Layer: {name}, Max Abs Value: {max_abs_value}")
                if prefix not in prefix_layers:
                    prefix_layers[prefix] = []
                prefix_layers[prefix].append((name, param))
        elif mode == 'original':
            if 'blocks' in name:
                prefix = name.split('.')[1]
                max_abs_value = param.abs().max().item()
                logger.info(f"Layer: {name}, Max Abs Value: {max_abs_value}")
                if prefix not in prefix_layers:
                    prefix_layers[prefix] = []
                prefix_layers[prefix].append((name, param))

    for prefix, layers in prefix_layers.items():
        plt.figure(figsize=(10, 6))
        for name, param in layers:
            param_values = param.detach().cpu().numpy().flatten()
            plt.plot(param_values, label=name)

        plt.title(f'Weights Visualization for {prefix}')
        plt.xlabel('Index')
        plt.ylabel('Weight Value')
        plt.legend()
        plot_path = os.path.join(save_dir, f'{prefix}_weights.png')
        plt.savefig(plot_path)
        plt.close()


def init_env(args, dir=None):
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    if dir:
        experiment_dir = f"{args.results_dir}/{dir}"
    else:
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
    return rank, device, logger, experiment_dir


def init_data_and_model(args, rank, device, logger):
    loader, sampler = init_data(args, rank, logger)
    model, state_dict, diffusion, vae = init_model(args, device)
    return loader, model, state_dict, diffusion, vae


def init_data(args, rank, logger):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    return loader, sampler


def init_model(args, device):
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    diffusion = create_diffusion(timestep_respacing=str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    return model, state_dict, diffusion, vae


def get_pq_model(model, file_path, rank, experiment_dir, logger, mode='train', type='new'):
    if mode == 'train':
        default_config = os.path.join(file_path, "../pqf/config/train_dit.yaml")
    elif mode == 'val':
        default_config = os.path.join(file_path, "../pqf/config/train_dit_val.yaml")
    if rank == 0:
        shutil.copy(default_config, experiment_dir)
    config = load_config_clean(file_path, default_config_path=default_config)
    model_config = config["model"]
    compression_config = model_config["compression_parameters"]
    uncompressed_model_size_bits = compute_model_nbits(model)
    model = compress_model(model, **compression_config, logger=logger, type=type).cuda()

    compressed_model_size_bits = compute_model_nbits(model)
    logger.info(f"Uncompressed model size: {uncompressed_model_size_bits} bits")
    logger.info(f"Compressed model size: {compressed_model_size_bits} bits")
    logger.info(f"Compression ratio: {uncompressed_model_size_bits / compressed_model_size_bits:.2f}")

    return model


def log_compare_weights(model_ori, model_comp, compress_mode, logger):
    fc_params = {
        1: ("fc1", "mlp", "fc1", "fc1_u", "fc1_v"),
        2: ("fc2", "mlp", "fc2", "fc2_u", "fc2_v"),
        3: ("qkv", "attn", "qkv", "qkv_u", "qkv_v"),
        4: ("proj", "attn", "proj", "proj_u", "proj_v"),
        5: ("adaln", "adaLN_modulation", "1", "1", "2")
    }
    with torch.no_grad():
        for block_id in range(len(model_comp.blocks)):
            block_comp = model_comp.blocks[block_id]
            block_ori = model_ori.blocks[block_id]
            for fc_idx in range(1, 6):
                file_name, layer_name, org_name, w_u_name, w_v_name = fc_params[fc_idx]
                weight_org = getattr(getattr(block_ori, layer_name), org_name).weight
                if compress_mode == 'pq':
                    weight_comp = getattr(getattr(block_comp, layer_name), org_name)._get_uncompressed_weight()
                elif compress_mode == 'uv':
                    weight_u = getattr(getattr(block_comp, layer_name), w_u_name).weight
                    weight_v = getattr(getattr(block_comp, layer_name), w_v_name).weight
                    weight_comp = torch.matmul(weight_v, weight_u)
                elif compress_mode == 'both':
                    weight_u = getattr(getattr(block_comp, layer_name), w_u_name)._get_uncompressed_weight()
                    weight_v = getattr(getattr(block_comp, layer_name), w_v_name)._get_uncompressed_weight()
                    weight_comp = torch.matmul(weight_v, weight_u)
                error = (weight_comp - weight_org).abs().mean().item()
                ori_abs = weight_org.abs().mean().item()
                logger.info(f"[Block({block_id})-Layer({file_name})] Ori abs: {ori_abs:.4f}, L1 Error: {error:.4f}")
    return



def log_params(model_before, model_after, logger):
    uncompressed_model_size_bits = compute_model_nbits(model_before)
    compressed_model_size_bits = compute_model_nbits(model_after)
    logger.info(f"Uncompressed model size: {uncompressed_model_size_bits} bits")
    logger.info(f"Compressed model size: {compressed_model_size_bits} bits")
    logger.info(f"Compression ratio: {uncompressed_model_size_bits / compressed_model_size_bits:.2f}")
    return

def parse_option():
    parser = argparse.ArgumentParser('DiT script', add_help=False)

    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")

    parser.add_argument("--data-path", type=str, default="./sampled_images")
    parser.add_argument("--results-dir", type=str, default="results/low_rank")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)

    parser.add_argument("--s2-mode", type=str, default="mse")
    parser.add_argument("--s3-mode", type=str, default="sample")
    parser.add_argument("--pq-mode", type=str, default="sample")
    parser.add_argument("--low-rank", action="store_true")
    parser.add_argument("--percent", type=float, default=0.9)
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--low-rank-ckpt", type=str, default=None)
    parser.add_argument("--pq", action="store_true")
    parser.add_argument("--pq-ckpt", type=str, default=None)
    parser.add_argument("--qwerty", action="store_true")
    parser.add_argument("--qwerty-ckpt", type=str, default=None)
    parser.add_argument("--qwerty-mode", type=str, default="distill")
    parser.add_argument("--opt-pq", action="store_true")


    args = parser.parse_args()

    return args