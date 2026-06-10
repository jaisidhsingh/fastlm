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

# Job specific vars
config=$1
job_idx=$2 # CONDOR job arrays range from 0 to n-1
job_cluster=$3

mkdir -p /fast/jsingh/tmp/mp/${job_cluster}/${job_idx}
mkdir -p /fast/jsingh/tmp/wandb/${job_cluster}/${job_idx}
mkdir -p /fast/jsingh/tmp/triton/${job_cluster}/${job_idx}
mkdir -p /fast/jsingh/tmp/inductor/${job_cluster}/${job_idx}

export TMPDIR=/fast/jsingh/tmp/mp/${job_cluster}/${job_idx}
export WANDB_CACHE_DIR=/fast/jsingh/tmp/wandb/${job_cluster}/${job_idx}
export TRITON_CACHE_DIR=/fast/jsingh/tmp/triton/${job_cluster}/${job_idx}
export TORCHINDUCTOR_CACHE_DIR=/fast/jsingh/tmp/inductor/${job_cluster}/${job_idx}

# Make `torch.compile` happy for GDN implementation
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
# Execute python script
python -m experiments.train --config=$config --job_idx=$job_idx --job_cluster=$job_cluster
