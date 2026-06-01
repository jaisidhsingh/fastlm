#!/bin/bash

source ~/.bashrc
echo $BASHRC_SRC_CHECK

source /lustre/fast/fast/jsingh/envs/miniconda3/etc/profile.d/conda.sh
conda activate pt
echo "Checking if environment is indeed on..."
pip show torch
echo " "
echo "Conda profile sourced and environment activated"

cd /home/jsingh/projects/fastlm
echo "setup done"

nvidia-smi

# Job specific vars
config=$1
job_idx=$2 # CONDOR job arrays range from 0 to n-1

# Execute python script
python3 -m experiments.train --config=$config --job_idx=$job_idx
