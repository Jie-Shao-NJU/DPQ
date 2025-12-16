import os
import argparse
import sys
sys.path.append("/mnt/petrelfs/shaojie/code/DiT/")
import torch
from copy import deepcopy
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from distributed import init_distributed_mode
from torch.nn.parallel import DistributedDataParallel as DDP

from models import DiT_models
from pq.utils_model import init_env, init_model, get_pq_model, cleanup, parse_option
from pq.utils_traineval import sample, train, dit_generator, save_ckpt

def main(args):
    init_distributed_mode(args)
    rank, device, logger, experiment_dir = init_env(args, dir='tmp')
    model, state_dict, diffusion, vae = init_model(args, device)
    checkpoint_dir = f"{experiment_dir}/checkpoints-train"  # Stores saved model checkpoints
    latent_size = args.image_size // 8

    if args.pq_ckpt == None:
        file_path = os.path.dirname(__file__)
        model_pq = get_pq_model(deepcopy(model), file_path, rank, experiment_dir, logger, mode='train')
        logger.info("PQ model done!")
        save_ckpt(model_pq, args, checkpoint_dir, logger)
    else:
        file_path = os.path.dirname(__file__)
        model_pq = get_pq_model(deepcopy(model), file_path, rank, experiment_dir, logger, mode='val', type='new')
        checkpoint = torch.load(args.pq_ckpt)
        state_dict = checkpoint['model']
        # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model_pq.load_state_dict(state_dict)

        # model_pq2 = get_pq_model(deepcopy(model), file_path, rank, experiment_dir, logger, mode='val', type='new')
        # checkpoint2 = torch.load("results/train_pq/010-DiT-XL-2/checkpoints/ckpt.pt")
        # state_dict2 = checkpoint2['model']
        # model_pq2.load_state_dict(state_dict2)

        # model_pq3 = get_pq_model(deepcopy(model), file_path, rank, experiment_dir, logger, mode='val', type='old')
        # checkpoint3 = torch.load("results/train_pq/001-DiT-XL-2/checkpoints/0005000.pt")
        # state_dict3 = checkpoint3['model']
        # state_dict3 = {k.replace('module.', ''): v for k, v in state_dict3.items()}
        # model_pq3.load_state_dict(state_dict3)

        # model_pq4 = get_pq_model(deepcopy(model), file_path, rank, experiment_dir, logger, mode='val', type='new')
        # checkpoint4 = torch.load("results/train_pq/005-DiT-XL-2/checkpoints-train-new/0100000.pt")
        # state_dict4 = checkpoint4['model']
        # model_pq4.load_state_dict(state_dict4)

    mode = args.pq_mode
    if mode == "sample":
        model_pq.eval()
        model_string_name = args.model.replace("/", "-")
        ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
        folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                    f"cfg-{args.cfg_scale}-seed-{args.global_seed}-100000-step100"
        sample_folder_dir = f"{experiment_dir}/{folder_name}"
        sample(args, model_pq, vae, diffusion, sample_folder_dir)
    elif mode == "gen":
        # model_pq.eval()
        model.eval()
        # model_pq1.eval()
        # model_pq2.eval()
        # model_pq3.eval()
        # model_pq4.eval()
        diffusion_gen = dit_generator(str(args.num_sampling_steps), latent_size=latent_size, device=device)
        diffusion_gen.forward_val(vae, [model.forward, model_pq.forward], cfg=False, name=f"{experiment_dir}", logger=logger)
        # diffusion_gen.forward_val(vae, [model.forward, model_pq3.forward, model_pq4.forward], cfg=False, name="tmp", logger=logger)

    elif mode == "train":
        model_pq = DDP(model_pq.to(device), device_ids=[rank])
        model_pq.train()
        train(args, logger, model_pq, vae, diffusion, checkpoint_dir)

    logger.info("Done!")
    # cleanup()


if __name__ == "__main__":
    args = parse_option()
    main(args)