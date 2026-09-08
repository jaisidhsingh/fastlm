#!/bin/bash

set -euo pipefail

nvidia-smi
module load cuda/12.9
nvcc --version

cd "/lustre/home/jsingh/projects/fastlm"

ATTN_06B_CONFIG="/lustre/home/jsingh/projects/fastlm/src/config/throughput/attn_0.6B.yaml"
ATTN_1B_CONFIG="/lustre/home/jsingh/projects/fastlm/src/config/throughput/attn_1B.yaml"

MODEL_SIZE="${1:-0.6B}"
GRAD_ACCUMULATION_STEPS="${2:-}"
gas_args=()
if [[ -n "$GRAD_ACCUMULATION_STEPS" ]]; then
  gas_args=(--grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS")
fi
case "$MODEL_SIZE" in
  0.6B) config="$ATTN_06B_CONFIG" ;;
  1B) config="$ATTN_1B_CONFIG" ;;
  *) echo "Usage: $0 [0.6B|1B] [GAS]" >&2; exit 2 ;;
esac

torchrun --nnodes=1 --nproc_per_node=4 -m experiments.measure_throughput \
  --config "$config" \
  --backend "fla" \
  --cluster_id "mpi" \
  --use_flex "no" \
  --use_intra_doc_masking "no" \
  "${gas_args[@]}"
