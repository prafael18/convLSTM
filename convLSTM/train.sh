#!/usr/bin/env bash

python3.5 train.py --machine=1 --gpu=.75 --color_space=lab --norm_type=fs --norm_dim=fnorm | tee -a stdout.txt
