#!/bin/bash

source /home/jasi149i/.bashrc
source /data/horse/ws/jasi149i-fastlm/envs/pt/bin/activate
cd /projects/p_neurasearch/fastlm

nvidia-smi
module load CUDA/13.0.0
nvcc --version

CONFIG=$1
SLURM_ARRAY_TASK_ID=$2
SLURM_JOB_ID=$3

mp_cache="/tmp/mp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
wandb_cache="/tmp/wandb/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
triton_cache="/tmp/triton/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
inductor_cache="/tmp/inductor/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p $mp_cache
mkdir -p $wandb_cache
mkdir -p $triton_cache
mkdir -p $inductor_cache

export TMPDIR=$mp_cache
export WANDB_CACHE_DIR=$wandb_cache
export TRITON_CACHE_DIR=$triton_cache
export TORCHINDUCTOR_CACHE_DIR=$inductor_cache

# Execute python script
cd /projects/p_neurasearch/fastlm
python -m experiments.train \
  --config=$CONFIG \
  --job_idx=$SLURM_ARRAY_TASK_ID \
  --job_cluster=$SLURM_JOB_ID;

rm -rf $mp_cache $wandb_cache $triton_cache $inductor_cache
