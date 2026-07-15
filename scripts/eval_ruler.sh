#!/bin/bash


arch_id=("attn" "gdn+attn_1-3" "gdn+attn_1-1" "gdn+attn_3-1" "gdn")
n="150M"
d="3.0B"
lr=0.001
gbs=32
cluster_id="mpi"


for aid in "${arch_id[@]}"; do
  ckpt_path=$(python -m services.download.checkpoint --arch_id=$aid --n=$n --d=$d --gbs=$gbs --lr=$lr)
  python -m experiments.eval.ruler \
    --arch_id=$aid \
    --n=$n \
    --gbs=$gbs \
    --lr=$lr \
    --ckpt_path=$ckpt_path \
    --cluster_id=$cluster_id
done
