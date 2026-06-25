#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --account=p_neurasearch
#SBATCH --job-name=tokenize_nemo
#SBATCH --output=/data/horse/ws/jasi149i-fastlm/logs/june/out/job-%j.out
#SBATCH --error=/data/horse/ws/jasi149i-fastlm/logs/june/err/job-%j.err

source /home/jasi149i/.bashrc
source /data/walrus/ws/jasi149i-hybridlms/envs/pt/bin/activate

nproc=32

cd /projects/p_neurasearch/fastlm

# tokenize already downloaded dataset (subsample of ~50B tokens)
python3 -m experiments.download_or_tokenize_data \
  --out_path="/data/horse/ws/jasi149i-fastlm/data/nemotron-cc-sample-mtsynth" \
  --cache_path="/data/horse/ws/jasi149i-fastlm/hf_cache" \
  --chunk \
  --tokenizer="gpt2" \
  --n_workers=$nproc \
  --seq_length=2048 --split_train_valid \
  --n_tokens_valid=10000000 \
  --dataset_path="MultiSynt/Nemotron-CC-sample-2" \
  --dataset_split="train"

  # --nrows_tokenize=75000000 \
  # --save_tokenized --save_tokenizer \
