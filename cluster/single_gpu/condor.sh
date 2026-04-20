#!/bin/bash

source /home/jsingh/.bashrc
conda activate pt
cd /home/jsingh/projects/fastlm

# Job specific vars
config=$1
job_idx=$2 # CONDOR job arrays range from 0 to n-1

# Execute python script
python3 -m experiments.train --config=$config --job_idx=$job_idx
