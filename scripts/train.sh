#!/bin/bash

config="src/config/attn_36M.yaml"

cd /home/jsingh/projects/fastlm

python3 -m experiments.train --config=$config;
