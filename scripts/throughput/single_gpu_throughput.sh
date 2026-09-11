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

ATTN_06B_CONFIG="$PROJECT/src/config/throughput/attn_0.6B.yaml"
ATTN_1B_CONFIG="$PROJECT/src/config/throughput/attn_1B.yaml"
GDN_06B_CONFIG="$PROJECT/src/config/throughput/gdn_0.6B.yaml"
GDN_1B_CONFIG="$PROJECT/src/config/throughput/gdn_1B.yaml"

MODEL_SIZE="${1:-0.6B}"
GRAD_ACCUMULATION_STEPS="${2:-}"
ARCH_ID="${3:-attn}"
gas_args=()
if [[ -n "$GRAD_ACCUMULATION_STEPS" ]]; then
  gas_args=(--grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS")
fi
case "$ARCH_ID/$MODEL_SIZE" in
  attn/0.6B) config="$ATTN_06B_CONFIG" ;;
  attn/1B) config="$ATTN_1B_CONFIG" ;;
  gdn/0.6B) config="$GDN_06B_CONFIG" ;;
  gdn/1B) config="$GDN_1B_CONFIG" ;;
  *) echo "Usage: $0 [0.6B|1B] [GAS] [attn|gdn]" >&2; exit 2 ;;
esac

python -m experiments.measure_throughput \
  --config "$config" \
  --backend "legacy" \
  --cluster_id "$CLUSTER_ID" \
  --use_flex "yes" \
  --use_intra_doc_masking "yes" \
  "${gas_args[@]}"
