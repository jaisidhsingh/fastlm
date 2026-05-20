#!/bin/bash

mkdir -p /fast/jsingh/tmp

# tokenize already download dataset (only to 1B tokens)
PYTHONPATH=. python3 src/data/datasets/prepare.py \
  --out_path="/fast/jsingh/data/nemotron-cc-sample-mtsynth" \
  --cache_path="/fast/jsingh/tmp" \
  --tokenize --chunk \
  --nrows_tokenize=1000000 \
  --save_tokenized --save_tokenizer \
  --seq_length=4096 --split_train_valid \
  --n_tokens_valid=10000000 \
  --dataset_path="MultiSynt/Nemotron-CC-sample-2" \
  --dataset_split="train"
