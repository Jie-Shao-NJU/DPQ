#!/bin/bash
set -x

CONFIG=${1}

# GPUS=${2:-40}
# GPUS_PER_NODE=${3:-8}
# GPUS=${2:-8}
# GPUS_PER_NODE=${3:-8}
GPUS=${2:-1}
GPUS_PER_NODE=${3:-1}
PARTITION=${4:-"INTERN2"}
QUOTA_TYPE=${5:-"reserved"}
JOB_NAME=${6:-"vl_sj"}

CPUS_PER_TASK=${CPUS_PER_TASK:-10}

if [ $GPUS -lt 8 ]; then
    NODES=1
else
    NODES=$((GPUS / GPUS_PER_NODE))
fi

# SRUN_ARGS=${SRUN_ARGS:-" --jobid=3756013"} # 3755917 3741547 3756013

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=32425    
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QUANT_FLAGS="--image-size 256 --ckpt pretrained_models/DiT-XL-2-256x256.pt --pq-mode gen --results-dir results/train_pq --epochs 10 --ckpt-every 5000 --data-path /mnt/petrelfs/share/images/train --global-batch-size 64
--pq-ckpt results/train_pq/005-DiT-XL-2/checkpoints-train-new/0100000.pt"

SAMPLE_FLAGS="--num-fid-samples 10000 --num-sampling-steps 250 --cfg-scale 1.5 --image-size 256"

srun -p ${PARTITION} \
  --job-name=${JOB_NAME} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  python pq/pq_getmodel.py $QUANT_FLAGS $SAMPLE_FLAGS
  # python evaluator.py samples/VIRTUAL_imagenet256_labeled.npz results/train_pq/005-DiT-XL-2/DiT-XL-2-DiT-XL-2-256x256-size-256-vae-mse-cfg-1.5-seed-0-100000-step50.npz
