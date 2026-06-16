#!/bin/bash

cd /home/jsingh/projects/fastlm

arch_id="gdn"
n=("20M" "50M")
gbs=(16 32)
lr="all_parallel"
mode="main"
bid=250

for psid in "${n[@]}"; do
  for glbs in "${gbs[@]}"; do
    python -m manager \
      --arch_id $arch_id \
      --n "$psid" \
      --gbs "$glbs" \
      --lr $lr \
      --mode $mode \
      --bid $bid;
  done
done

