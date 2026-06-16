#!/bin/bash

cd /home/jsingh/projects/fastlm

arch_id="gdn"
n=("20M")
gbs=(16)
lr="all_parallel"
mode="main"
bid=250
submit="yes"

for psid in "${n[@]}"; do
  for glbs in "${gbs[@]}"; do
    python -m manager \
      --arch_id $arch_id \
      --n "$psid" \
      --gbs "$glbs" \
      --lr $lr \
      --mode $mode \
      --bid $bid \
      --submit $submit;
  done
done
