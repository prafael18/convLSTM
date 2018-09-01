#!/usr/bin/env bash

#python3 train_reg.py --machine=2 --gpu=1.0 --color_space=rgb --norm_dim=raw -e 50 -l 1e-3 >> stdout.txt 2>&1
#python3.6 train_reg.py --machine=1 --gpu=1.0 --color_space=rgb --norm_dim=raw -e 51 -l 1e-4 
#python3.6 train_reg.py --machine=1 --gpu=1.0 --color_space=rgb --norm_dim=raw -e 52 -l 5e-4
#python3.6 train_reg.py --machine=1 --gpu=1.0 --color_space=rgb --norm_dim=raw -e 53 -l 5e-3

python3.6 train.py --machine=1 --gpu=1.0 --color_space=lab --norm_dim=raw -e 60
