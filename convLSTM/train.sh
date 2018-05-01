#!/usr/bin/env bash

./train.py --machine=0 --gpu=0.75 --color_space=rgb --norm_type=raw | tee -a stdout.txt
