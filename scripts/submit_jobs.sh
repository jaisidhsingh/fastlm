#!/bin/bash

cd /home/jsingh/projects/fastlm

arch_id="attn"
n=("150M" "300M")
gbs=(16 32)
lr="all_parallel"
mode="decay"
bid=500

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

