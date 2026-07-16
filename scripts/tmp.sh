#!/bin/bash

# Job specific vars
bench="ruler"
config="/home/jsingh/projects/fastlm/execs/gdn+attn_3-1/150M/eval-job_gbs-32_lr-0p001_d-all_parallel.yaml"
job_idx=0 # CONDOR job arrays range from 0 to n-1
job_cluster=1292884
cluster_id="mpi"

mp_cache="/fast/jsingh/tmp/mp/${job_cluster}/${job_idx}"
mkdir -p /fast/jsingh/tmp/mp/${job_cluster}/${job_idx}
export TMPDIR=/fast/jsingh/tmp/mp/${job_cluster}/${job_idx}

ckpt_path=$(python -m services.download.checkpoint --config $config --job_idx $job_idx --job_cluster $job_cluster --cluster_id $cluster_id)

python -m experiments.eval.${bench} \
  --config=$config \
  --ckpt_path=$ckpt_path \
  --job_idx=$job_idx \
  --job_cluster=$job_cluster \
  --cluster_id=$cluster_id

rm -rf mp_cache
