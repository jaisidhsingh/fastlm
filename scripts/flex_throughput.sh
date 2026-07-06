#!/bin/bash

cd /lustre/home/jsingh/projects/fastlm

config="/lustre/home/jsingh/projects/fastlm/src/config/throughput/attn_150M.yaml"

# full, math
python -m experiments.measure_throughput \
  --config $config \
  --use_flex "no" \
  --use_intra_doc_masking "no"
# full, best
python -m experiments.measure_throughput \
  --config $config \
  --use_flex "yes" \
  --use_intra_doc_masking "no"

# intra-doc, math
python -m experiments.measure_throughput \
  --config $config \
  --use_flex "no" \
  --use_intra_doc_masking "yes"
# intra-doc, best
python -m experiments.measure_throughput \
  --config $config \
  --use_flex "yes" \
  --use_intra_doc_masking "yes"
