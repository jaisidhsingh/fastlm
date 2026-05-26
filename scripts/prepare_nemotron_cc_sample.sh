!/bin/bash

mkdir -p /fast/jsingh/tmp
mkdir -p /fast/jsingh/logs/fastlm/tokenization
mkdir -p /fast/jsingh/logs/fastlm/tokenization/err
mkdir -p /fast/jsingh/logs/fastlm/tokenization/log
mkdir -p /fast/jsingh/logs/fastlm/tokenization/out

# tokenize already downloaded dataset (subsample of ~50B tokens)
PYTHONPATH=/home/jsingh/projects/fastlm python3 src/data/datasets/prepare.py \
  --out_path="/fast/jsingh/data/nemotron-cc-sample-mtsynth" \
  --cache_path="/fast/jsingh/tmp" \
  --tokenize --chunk \
  --nrows_tokenize=75000000 \
  --save_tokenized --save_tokenizer \
  --seq_length=2048 --split_train_valid \
  --n_tokens_valid=10000000 \
  --dataset_path="MultiSynt/Nemotron-CC-sample-2" \
  --dataset_split="train"
