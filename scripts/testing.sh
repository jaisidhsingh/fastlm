#!/bin/bash

cd /home/jsingh/projects/fastlm

config="src/config/attn/small_scale/attn_10M.yaml"
steps=50

python -m experiments.find_max_mbs --config=$config --steps=$steps
