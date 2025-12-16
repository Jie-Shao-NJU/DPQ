#!/bin/bash
set -x

CONFIG=${1}

GPUS=${2:-1}
GPUS_PER_NODE=${3:-1}
PARTITION=${4:-"INTERN3"}
QUOTA_TYPE=${5:-"reserved"}
JOB_NAME=${6:-"vl_sj"}

CPUS_PER_TASK=${CPUS_PER_TASK:-10}

if [ $GPUS -lt 8 ]; then
    NODES=1
else
    NODES=$((GPUS / GPUS_PER_NODE))
fi

# SRUN_ARGS=${SRUN_ARGS:-" --jobid=3703936"} # 3768157 3768158 3789766 -w HOST-10-140-66-41  3636795

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=32423
# export NCCL_DEBUG=INFO
# export TF_CPP_MIN_LOG_LEVEL=3
# unset CUDA_LAUNCH_BLOCKING
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_LOGS="+dynamo" d
# export TORCHDYNAMO_VERBOSE=1
# export TORCH_DISTRIBUTED_DEBUG=INFO

# enable for sampling only
# export LD_LIBRARY_PATH="/mnt/petrelfs/share_data/tianchangyao.p/cuda/cuda-11.7/lib64":$LD_LIBRARY_PATH

srun -p ${PARTITION} \
  --job-name=${JOB_NAME} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  -w SH-IDCA1404-10-140-54-107 \
  ${SRUN_ARGS} \
  python pq/main_dit_calkl_s2_2.py --image-size 256 --ckpt pretrained_models/DiT-XL-2-256x256.pt --global-batch-size 8  