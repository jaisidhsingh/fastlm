#!/bin/bash

config="src/config/hybrid/adamw/hybrid_reverse_36M.yaml"

cd /home/jsingh/projects/fastlm

export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
python3 -m experiments.train --config=$config;
