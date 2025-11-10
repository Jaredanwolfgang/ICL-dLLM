#!/bin/bash

set -x

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=6,7

NNODES=${NNODES:=1}
NPROC_PER_NODE=2
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}

torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT tasks/train_torch.py \
  configs/pretrain/qwen2_5_coder_500M.yaml \
  --model.model_path=/mnt/Qwen2.5-Coder-0.5B \
  --model.tokenizer_path=/mnt/Qwen2.5-Coder-0.5B \
  --data.train_path=/mnt/fine_code/data \
  --train.save_steps=100