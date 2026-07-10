#!/bin/bash

nvidia-smi
module load cuda/12.9
nvcc --version

cd /lustre/home/jsingh/projects/fastlm

config="/lustre/home/jsingh/projects/fastlm/src/config/throughput/gdn_300M.yaml"
# config="/lustre/home/jsingh/projects/fastlm/src/config/throughput/attn_1mB.yaml"

torchrun --nnodes=1 --nproc_per_node=4 -m experiments.measure_throughput \
  --config $config \
  --use_flex "yes" \
  --use_intra_doc_masking "yes"
