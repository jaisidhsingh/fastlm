#!/bin/bash

cd /home/jsingh/projects/fastlm

arch_id="attn"
param_scale_ids=("50M" "150M")
global_batch_sizes=(16 32 64 128 256 512)
steps=10

for param_scale_id in "${param_scale_ids[@]}"; do
  for gbs in "${global_batch_sizes[@]}"; do
    python -m experiments.throughput_analysis \
      --arch_id=$arch_id \
      --param_scale_id=$param_scale_id \
      --global_batch_size=$gbs \
      --steps=$steps;
  done
done
