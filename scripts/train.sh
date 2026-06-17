#!/bin/bash

# source ~/.bashrc
# echo $BASHRC_SRC_CHECK

# source /lustre/fast/fast/jsingh/envs/miniconda3/etc/profile.d/conda.sh
# conda activate pt
# echo "Checking if environment is indeed on..."
# echo " "
# pip show torch
# echo " "
# echo "Conda profile sourced and environment activated"

# cd /home/jsingh/projects/fastlm
# echo "setup done"

# nvidia-smi

# Job specific vars
config='/lustre/home/jsingh/projects/fastlm/june_exec/decay_intermediates/cfg-decay_gbs-32_lr-0p002_n-300M.yaml'


# Make `torch.compile` happy for GDN implementation
# export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=0
# export TORCH_LOGS="+inductor,+dynamo,+aot"
# export TORCHDYNAMO_VERBOSE=1
# export TORCHINDUCTOR_CPP_WRAPPER=0
# Execute python script
python -m experiments.train --config=$config
