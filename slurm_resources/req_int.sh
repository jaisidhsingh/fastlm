#!/bin/bash

salloc --nodes=2 --ntasks=2 --cpus-per-task=6 \
       --gres=gpu:1 \
       --time=01:00:00 \
       --mem=32G \
       --account=p_neurasearch
