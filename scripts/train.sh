#!/bin/bash

config="src/config/attn/adamw/attn_live.yaml"

cd /home/jsingh/projects/fastlm

# export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
python3 -m experiments.train --config=$config;
