#!/bin/bash

config="/lustre/home/jsingh/projects/fastlm/execs/attn/300M/tmp.yaml"

torchrun --nproc_per_node=2 -m experiments.train --config=$config
