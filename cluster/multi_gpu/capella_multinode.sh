#!/bin/bash

source /home/jasi149i/.bashrc
conda activate /home/jasi149i/.conda/envs/pt
echo "Check if environment is indeed on"
pip show torch
cd /projects/p_neurasearch/fastlm

nvidia-smi
module load cuda/13
nvcc --version

CONFIG=$1
SLURM_ARRAY_TASK_ID=$2
SLURM_JOB_ID=$3
DP=$4
cluster_id="capella"

mp_cache="/data/horse/ws/jasi149i/tmp/mp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
wandb_cache="/data/horse/ws/jasi149i/tmp/wandb/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
triton_cache="/data/horse/ws/jasi149i/tmp/triton/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
inductor_cache="/data/horse/ws/jasi149i/tmp/inductor/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "mp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p wandb/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p triton/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p inductor/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

export TMPDIR=$mp_cache
export WANDB_CACHE_DIR=$wandb_cache
export TRITON_CACHE_DIR=$triton_cache
export TORCHINDUCTOR_CACHE_DIR=$inductor_cache

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((15000 + ((SLURM_JOB_ID * 131 + SLURM_ARRAY_TASK_ID) % 45000)))

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "SLURM_NODEID=${SLURM_NODEID}"
echo "SLURM_NNODES=${SLURM_NNODES}"

cd /projects/p_neurasearch/fastlm

srun torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc-per-node=4 \
    --node-rank="${SLURM_NODEID}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    -m experiments.train \
    --config="${CONFIG}" \
    --job_idx="${SLURM_ARRAY_TASK_ID}" \
    --job_cluster="${SLURM_JOB_ID}" \
    --cluster_id="${cluster_id}"

rm -rf $mp_cache $wandb_cache $triton_cache $inductor_cache
