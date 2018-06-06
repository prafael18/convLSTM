#!/usr/bin/env bash

python3 train.py --machine=3 --gpu=.75 --color_space=rgb --norm_dim=raw | tee -a stdout.txt
