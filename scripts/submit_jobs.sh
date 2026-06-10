#!/bin/bash

cd /home/jsingh/projects/fastlm

arch_id="attn"
n="20M"
gbs=32
lr="all_parallel"
mode="decay"
bid=1000

python -m manager --arch_id=$arch_id --n=$n --gbs=$gbs --lr=$lr --mode=$mode --bid=$bid
