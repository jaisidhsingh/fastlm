#!/bin/bash

mkdir -p /fast/jsingh/tmp

PYTHONPATH=. python3 src/data/datasets/prepare.py \
  --out_path="/fast/jsingh/data/nemotron-cc-sample-mtsynth" \
  --cache_path="/fast/jsingh/tmp" \
  --download --save_raw \
  --dataset_path="MultiSynt/Nemotron-CC-sample-2" \
  --dataset_split="train"
