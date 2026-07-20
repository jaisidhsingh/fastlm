#!/bin/bash


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

echo $CLUSTER_ID $PROJECT
cd $PROJECT

module load cuda/12.4
nvidia-smi
nvcc --version

CONFIG=$PROJECT"/src/config/int/hybrid_3-1_300M.yaml"
DP=$1

export TORCHDYNAMO_VERBOSE=1

torchrun --nnodes=1 --standalone --nproc_per_node=$DP -m experiments.train \
  --config=$CONFIG \
  --cluster_id=$CLUSTER_ID;
