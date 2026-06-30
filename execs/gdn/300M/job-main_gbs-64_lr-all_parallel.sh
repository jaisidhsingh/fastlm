#!/bin/bash
#SBATCH --job-name=alphafastlm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --account=p_neurasearch
#SBATCH --job-name=gdn-main_n-300M_gbs-64_lr-all_parallel
#SBATCH --output=/data/horse/ws/jasi149i-fastlm/logs/june/out/job-%A_%a.out
#SBATCH --error=/data/horse/ws/jasi149i-fastlm/logs/june/err/job-%A_%a.err
#SBATCH --array=0-5
#SBATCH --exclude=i8009

CONFIG=/projects/p_neurasearch/alphafastlm/execs/gdn/300M/cfg-main_gbs-64_lr-all_parallel.yaml
DP=4
HOST=$(hostname -f)

cd /projects/p_neurasearch/alphafastlm

if [ "$DP" -eq 1 ]; then
    bash cluster/single_gpu/slurm.sh "$CONFIG" "$SLURM_ARRAY_TASK_ID" "$SLURM_JOB_ID"
else
  if [ "$HOST" == *alpha* && "$DP" -eq 16]; then
    bash cluster/multi_gpu/multinode_slurm.sh "$CONFIG" "$SLURM_ARRAY_TASK_ID" "$SLURM_JOB_ID" "$DP"
  else
    bash cluster/multi_gpu/slurm.sh "$CONFIG" "$SLURM_ARRAY_TASK_ID" "$SLURM_JOB_ID" "$DP"
  fi
fi
