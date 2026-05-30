#!/bin/bash

cd /home/jsingh/projects/fastlm

param_scale_id="50M"
steps=100

python -m experiments.throughput_analysis --param_scale_id=$param_scale_id --steps=$steps
