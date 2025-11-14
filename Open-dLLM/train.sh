#!/bin/bash

set -x

export TOKENIZERS_PARALLELISM=false
# export CUDA_VISIBLE_DEVICES=6,7

NNODES=${NNODES:=1}
NPROC_PER_NODE=4
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}

torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT tasks/train_icl.py \
  configs/pretrain/qwen2_5_coder_500M.yaml \
  --data.train_path=/mnt/fine_code/data \
  --data.data_name="gaussian" \
  --data.n_dims=20 \
  --train.save_steps=25000 \
  --train.task_name="linear_regression" \
  --train.max_steps=500001 \
  --train.curriculum='{"dims": {"start": 5, "end": 20, "inc": 1, "interval": 2000}, "points": {"start": 5, "end": 20, "inc": 1, "interval": 2000}}'
