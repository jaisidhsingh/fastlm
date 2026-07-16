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
bench=$1
config=$2
job_idx=$3 # CONDOR job arrays range from 0 to n-1
job_cluster=$4
cluster_id="mpi"

mp_cache="/fast/jsingh/tmp/mp/${job_cluster}/${job_idx}"
mkdir -p /fast/jsingh/tmp/mp/${job_cluster}/${job_idx}
export TMPDIR=/fast/jsingh/tmp/mp/${job_cluster}/${job_idx}

ckpt_path=$(python -m services.download.checkpoint --config $config)
python -m experiments.eval.${bench} \
  --config=$config \
  --ckpt_path=$ckpt_path \
  --job_idx=$job_idx \
  --job_cluster=$job_cluster \
  --cluster_id=$cluster_id

rm -rf mp_cache
