#!/bin/bash

config="src/config/hybrid/adamw/hybrid_reverse_36M.yaml"

cd /home/jsingh/projects/fastlm

python3 -m experiments.train --config=$config;
