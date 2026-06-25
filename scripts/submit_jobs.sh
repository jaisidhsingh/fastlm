#!/bin/bash

cd /home/jsingh/projects/fastlm

arch_id=("attn" "gdn")
n=("20M" "50M")
gbs=(16 32)
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
