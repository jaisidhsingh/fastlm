#!/bin/bash

cd /home/jsingh/projects/fastlm

arch_id="attn"
n=("20M" "50M")
gbs=(64 128)
lr="all_parallel"
mode="main"
bid=200
submit="yes"
routine="train"

for psid in "${n[@]}"; do
  for glbs in "${gbs[@]}"; do
    python -m manager \
      --arch_id $arch_id \
      --n "$psid" \
      --gbs "$glbs" \
      --lr $lr \
      --mode $mode \
      --bid $bid \
      --submit $submit \
      --routine $routine;
  done
done
