#!/bin/bash

source /home/jasi149i/.bashrc
# source /data/horse/ws/jasi149i-hybridlms/envs/pt/bin/activate
#module load Miniconda3/25.5.1-1
#source /software/genoa/r25.06/Miniconda3/25.5.1-1/etc/profile.d/conda.sh
conda activate ~/.conda/envs/pt
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

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

cluster_id="alpha"

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

# Execute python script
cd /projects/p_neurasearch/fastlm
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m experiments.train \
    --config=$CONFIG \
    --job_idx=$SLURM_ARRAY_TASK_ID \
    --job_cluster=$SLURM_JOB_ID \
    --cluster_id=$cluster_id;

rm -rf $mp_cache $wandb_cache $triton_cache $inductor_cache
