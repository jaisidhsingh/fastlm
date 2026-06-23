#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --account=p_neurasearch
#SBATCH --job-name=nemo_cc_mts_download
#SBATCH --output=/data/horse/ws/jasi149i-fastlm/logs/june/out/job-%j.out
#SBATCH --error=/data/horse/ws/jasi149i-fastlm/logs/june/err/job-%j.err

source /home/jasi149i/.bashrc
source /data/walrus/ws/jasi149i-hybridlms/envs/pt/bin/activate

nproc=16

python hf_ds_down.py $nproc
