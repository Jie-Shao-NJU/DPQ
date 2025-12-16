import torch
import time
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models

# 固定参数设置
model_name = "DiT-XL/2"
image_size = 256
vae_type = "mse"
num_classes = 1000
cfg_scale = 4.0
num_sampling_steps = 250
seed = 0
ckpt_path = "pretrained_models/DiT-XL-2-256x256.pt"  # 或设为自定义路径，如 "checkpoints/DiT-XL-2-256x256.pt"

def main():
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("📦 Loading DiT model and VAE...")
    start_load = time.time()

    latent_size = image_size // 8
    model = DiT_models[model_name](
        input_size=latent_size,
        num_classes=num_classes
    ).to(device)

    final_ckpt = ckpt_path or f"DiT-XL-2-{image_size}x{image_size}.pt"
    state_dict = find_model(final_ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}").to(device)
    diffusion = create_diffusion(str(num_sampling_steps))

    torch.cuda.synchronize()
    load_time = time.time() - start_load
    print(f"✅ Model + VAE loaded in {load_time:.2f} seconds")

    # Prepare latent noise and labels (only 1 image, duplicated for classifier-free guidance)
    z = torch.randn(1, 4, latent_size, latent_size, device=device)
    y = torch.tensor([417], device=device)  # Example label: golden retriever
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000], device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    # Measure sampling time and GPU memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_gen = time.time()

    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape, z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device
    )

    torch.cuda.synchronize()
    end_gen = time.time()
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Decode and save
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample
    save_image(samples, "profile_sample.png", normalize=True, value_range=(-1, 1))

    print(f"🕒 Sampling time: {end_gen - start_gen:.2f} seconds")
    print(f"📊 Peak GPU memory: {peak_mem:.2f} GB")

if __name__ == "__main__":
    main()
