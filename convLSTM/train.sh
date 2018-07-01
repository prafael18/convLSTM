#!/usr/bin/env bash

python3 train.py --machine=1 --gpu=.75 --color_space=rgb --norm_dim=raw | 2>&1 tee -a stdout.txt
