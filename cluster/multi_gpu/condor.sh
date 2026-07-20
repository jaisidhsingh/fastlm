#!/bin/bash

source ~/.bashrc
echo $BASHRC_SRC_CHECK

source /lustre/fast/fast/jsingh/envs/miniconda3/etc/profile.d/conda.sh
conda activate pt
echo "Checking if environment is indeed on..."
echo " "
pip show torch
echo " "
echo "Conda profile sourced and environment activated"

cd /home/jsingh/projects/fastlm
echo "setup done"

nvidia-smi
module load cuda/12.4
nvcc --version

# Job specific vars
config=$1
job_idx=$2 # CONDOR job arrays range from 0 to n-1
job_cluster=$3
dp=$4
cluster_id="mpi"

mp_cache="/fast/jsingh/tmp/mp/${job_cluster}/${job_idx}"
wandb_cache="/fast/jsingh/tmp/wandb/${job_cluster}/${job_idx}"
triton_cache="/fast/jsingh/tmp/triton/${job_cluster}/${job_idx}"
inductor_cache="/fast/jsingh/tmp/inductor/${job_cluster}/${job_idx}"

mkdir -p /fast/jsingh/tmp/mp/${job_cluster}/${job_idx}
mkdir -p /fast/jsingh/tmp/wandb/${job_cluster}/${job_idx}
mkdir -p /fast/jsingh/tmp/triton/${job_cluster}/${job_idx}
mkdir -p /fast/jsingh/tmp/inductor/${job_cluster}/${job_idx}

export TMPDIR=/fast/jsingh/tmp/mp/${job_cluster}/${job_idx}
export WANDB_CACHE_DIR=/fast/jsingh/tmp/wandb/${job_cluster}/${job_idx}
export TRITON_CACHE_DIR=/fast/jsingh/tmp/triton/${job_cluster}/${job_idx}
export TORCHINDUCTOR_CACHE_DIR=/fast/jsingh/tmp/inductor/${job_cluster}/${job_idx}


# standalone avoids having to specify this
# master_port=$((15000 + ((SLURM_JOB_ID * 131 + SLURM_ARRAY_TASK_ID) % 45000)))
# export MASTER_PORT=$master_port

# Execute python script
torchrun --nnodes=1 --standalone --nproc_per_node=$dp -m experiments.train \
  --config=$config \
  --job_idx=$job_idx \
  --job_cluster=$job_cluster \
  --cluster_id=$cluster_id;

rm -rf mp_cache wandb_cache triton_cache inductor_cache
