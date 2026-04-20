#!/bin/bash

config="src/config/hybrid_150M.yaml"

cd /home/jsingh/projects/fastlm

python3 -m experiments.train --config=$config;
