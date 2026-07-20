#!/bin/bash

source /home/jasi149i/.bashrc
conda activate ~/.conda/envs/pt
echo "Check if environment is indeed on"
pip show torch
cd /projects/p_neurasearch/fastlm

nvidia-smi
module load CUDA/13.0.0
nvcc --version

CONFIG=$1
SLURM_ARRAY_TASK_ID=$2
SLURM_JOB_ID=$3
DP=$4
cluster_id="alpha"

mp_cache="/tmp/mp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
wandb_cache="/tmp/wandb/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
triton_cache="/tmp/triton/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
inductor_cache="/tmp/inductor/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p $mp_cache $wandb_cache $triton_cache $inductor_cache

export TMPDIR=$mp_cache
export WANDB_CACHE_DIR=$wandb_cache
export TRITON_CACHE_DIR=$triton_cache
export TORCHINDUCTOR_CACHE_DIR=$inductor_cache

#master_port=$((15000 + ((SLURM_JOB_ID * 131 + SLURM_ARRAY_TASK_ID) % 45000)))
#export MASTER_PORT=$master_port

# Execute python script
cd /projects/p_neurasearch/fastlm
torchrun --nnodes=1 --standalone --nproc_per_node=$DP -m experiments.train \
  --config=$CONFIG \
  --job_idx=$SLURM_ARRAY_TASK_ID \
  --job_cluster=$SLURM_JOB_ID \
  --cluster_id=$cluster_id \

rm -rf $mp_cache $wandb_cache $triton_cache $inductor_cache
