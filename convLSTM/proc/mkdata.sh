#!/usr/bin/env bash

#python3 augment.py --dest_dir=augm
#python3 augment.py --rng --dest_dir=augm_rng

#python3 video2tfrecord.py -m 1 -s train -c rgb -n raw -f many -d raw_rgb_augm_rng


python3 video2tfrecord.py -m 3 -s train -c rgb -n raw -f many -d rgb_raw_tv
