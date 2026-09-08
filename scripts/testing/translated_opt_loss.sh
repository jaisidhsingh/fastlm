#!/bin/bash

# Load a legacy checkpoint into the FLA backend, weights and AdamW state, then
# train both backends for a few steps on the same batches and compare losses.
#
# The model config, the optimizer settings, the weights and the AdamW state all
# come out of the checkpoint, so nothing about the model is set here. Only the
# checkpoint to read and how long to run are.
#
# Usage:
#   bash scripts/testing/translated_opt_loss.sh
#   bash scripts/testing/translated_opt_loss.sh /path/to/ckpt.pt 5

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

# the GDN backward needs tilelang, and tilelang locates CUDA through nvcc
load_cuda() {
  case "$1" in
    mpi) module load cuda/12.9 ;;
    capella) module load CUDA/13.0.0 ;;
    alpha) module load CUDA/13.0.0 ;;
    *) echo "unknown cluster, not loading a CUDA module" ;;
  esac
}

CLUSTER_ID=$(detect_cluster)
PROJECT=$(find_project)

echo $CLUSTER_ID $PROJECT

cd $PROJECT

load_cuda $CLUSTER_ID
nvcc --version

# checkpoint to translate, and how many steps to train after loading it.
# gdn+attn_3-1 at the 50M specification, at the end of its decay to 3.0B tokens.
# The model config and the AdamW settings both come out of this file, so nothing
# about the model is set here.
ckpt_path="/fast/jsingh/projects/fastlm/june/results/gdn+attn_3-1/50M/gbs_wise_results/gbs_16/checkpoints/lr_0p001/ckpt_decayed_to_3p0B.pt"
steps=10
resume_step=0
batch_size=2
batch_seed=1

if [ ! -f "$ckpt_path" ]; then
  echo "checkpoint not found: $ckpt_path"
  exit 1
fi

python -m testing.translated_optimiser_loss \
  --ckpt-path "$ckpt_path" \
  --steps "$steps" \
  --resume-step "$resume_step" \
  --batch-size "$batch_size" \
  --batch-seed "$batch_seed"
