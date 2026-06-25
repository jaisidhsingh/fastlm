#!/bin/bash

cd /projects/p_neurasearch/fastlm

arch_id=("attn")
n=("20M")
gbs=(64)
lr="all_parallel"
mode="main"
submit="yes"
routine="train"

for aid in "${arch_id[@]}"; do
  for psid in "${n[@]}"; do
    for glbs in "${gbs[@]}"; do
      python -m manager \
        --arch_id $aid \
        --n "$psid" \
        --gbs "$glbs" \
        --lr $lr \
        --mode $mode \
        --submit $submit \
        --routine $routine;
    done
  done
done
