#!/bin/bash

set -euo pipefail

# tell us which cluster we're on
detect_cluster() {
  case "$(hostname -f)" in
    *cluster.is*) echo "mpi" ;;
    *capella*) echo "capella" ;;
    *alpha*) echo "alpha" ;;
    *)   echo "unknown" ;;
  esac
}

# map the path to our codebase each to available cluster
find_project() {
  case "$(hostname -f)" in
    *cluster.is*) echo "/home/jsingh/projects/fastlm" ;;
    *capella*) echo "/projects/p_neurasearch/fastlm" ;;
    *alpha*) echo "/projects/p_neurasearch/fastlm" ;;
    *)   echo "unknown" ;;
  esac
}

CLUSTER_ID=$(detect_cluster)
PROJECT=$(find_project)

echo "$CLUSTER_ID $PROJECT"

cd "$PROJECT"

MODEL_SIZE="${1:-0.6B}"
MICRO_BATCH_SIZE=8
GRAD_ACCUMULATION_STEPS=8

case "$MODEL_SIZE" in
  0.6B|1B) ;;
  *) echo "Usage: $0 [0.6B|1B]" >&2; exit 2 ;;
esac

for arch_id in attn gdn; do
  config="$PROJECT/src/config/throughput/${arch_id}_${MODEL_SIZE}.yaml"
  if [[ ! -f "$config" ]]; then
    echo "Missing config: $config" >&2
    exit 2
  fi
  for backend in fla legacy; do
    echo " "
    echo "=== ${arch_id} [${MODEL_SIZE}] on ${backend} backend, MBS=${MICRO_BATCH_SIZE}, GAS=${GRAD_ACCUMULATION_STEPS} ==="
    echo " "
    python -m experiments.measure_throughput \
      --config "$config" \
      --backend "$backend" \
      --cluster_id "$CLUSTER_ID" \
      --use_flex "yes" \
      --use_intra_doc_masking "yes" \
      --micro_batch_size "$MICRO_BATCH_SIZE" \
      --grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS"
  done
done
