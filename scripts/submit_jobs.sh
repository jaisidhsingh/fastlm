#!/bin/bash

cd /home/jsingh/projects/fastlm

arch_id=("attn")
n=("150M")
gbs=(64)
lr="all_parallel"
mode="decay"
bid=100
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
        --bid $bid \
        --submit $submit \
        --routine $routine;
    done
  done
done
