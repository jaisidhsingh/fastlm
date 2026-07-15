#!/bin/bash


arch_id="attn"
n="150M"
d="3.0B"
lr=0.001
gbs=32
cluster_id="mpi"

ckpt_path=$(python -m services.download.checkpoint --arch_id=$arch_id --n=$n --d=$d --gbs=$gbs --lr=$lr)
python -m experiments.eval.dclm_core --arch_id=$arch_id --n=$n --ckpt_path=$ckpt_path --cluster_id=$cluster_id
