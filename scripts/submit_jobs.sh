#!/bin/bash

cd /home/jsingh/projects/fastlm

arch_id="gdn"
n=("300M")
gbs=(32)
lr="all_parallel"
mode="decay"
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
