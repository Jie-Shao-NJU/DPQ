#!/bin/bash
set -x

CONFIG=${1}

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

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=32424    

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
  python speed.py
  # python evaluator.py $EVAL_FLAGS
