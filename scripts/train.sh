#!/bin/bash

cd /projects/p_neurasearch/fastlm
config='/projects/p_neurasearch/fastlm/execs/gdn+attn_3-1/20M/tmp.yaml'
python -m experiments.train --config=$config
