#!/bin/bash

source /home/jasi149i/.bashrc
conda activate ~/.conda/envs/pt
echo "Check if environment is indeed on"
pip show torch

cd /projects/p_neurasearch/fastlm

nvidia-smi

BENCH=$1
CONFIG=$2
SLURM_ARRAY_TASK_ID=$3
SLURM_JOB_ID=$4

cluster_id="capella"

mp_cache="/tmp/mp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p $mp_cache
export TMPDIR=$mp_cache

# Execute python script
ckpt_path=$(python -m services.download.checkpoint --config $CONFIG)
python -m experiments.eval.${BENCH} \
  --config=$CONFIG \
  --job_idx=$SLURM_ARRAY_TASK_ID \
  --job_cluster=$SLURM_JOB_ID \
  --cluster_id=$cluster_id;

rm -rf $mp_cache
