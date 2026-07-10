#!/bin/bash

cd /lustre/home/jsingh/projects/fastlm

config="/lustre/home/jsingh/projects/fastlm/src/config/throughput/gdn_300M.yaml"

python -m experiments.measure_throughput \
  --config $config \
  --use_flex "yes" \
  --use_intra_doc_masking "yes"
