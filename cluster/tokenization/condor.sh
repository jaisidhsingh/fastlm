#!/bin/bash

source ~/.bashrc
activate_pt
cd /projects/p_neurasearch/fastlm

# tokenize already downloaded dataset (subsample of ~50B tokens)
python3 -m experiments.download_or_tokenize_data \
  --out_path="/data/walrus/ws/jasi149i-hybridlms/data/nemotron-cc-sample-mtsynth" \
  --cache_path="/data/walrus/ws/jasi149i-hybridlms/tmp/hf" \
  --download --tokenize --chunk \
  --tokenizer="gpt2" \
  --nrows_tokenize=75000000 \
  --n_workers=16 \
  --save_tokenized --save_tokenizer \
  --seq_length=2048 --split_train_valid \
  --n_tokens_valid=10000000 \
  --dataset_path="MultiSynt/Nemotron-CC-sample-2" \
  --dataset_split="train"
