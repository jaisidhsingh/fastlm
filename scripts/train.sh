#!/bin/bash

cd /projects/p_neurasearch/alphafastlm
config='/projects/p_neurasearch/alphafastlm/execs/gdn+attn_3-1/20M/tmp.yaml'
python -m experiments.train --config=$config
