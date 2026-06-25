#!/bin/bash

cd /projects/p_neurasearch/fastlm
config='/projects/p_neurasearch/fastlm/execs/int/tmp.yaml'
python -m experiments.train --config=$config
