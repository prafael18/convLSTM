#!/usr/bin/env bash

#python3 augment.py --dest_dir=augm
#python3 augment.py --rng --dest_dir=augm_rng

#python3 video2tfrecord.py -m 1 -s train -c rgb -n raw -f many -d raw_rgb_augm_rng


#python3 video2tfrecord.py -m 3 -s train -c rgb -n raw -f many -d rgb_raw_tv

#python3.6 augment.py -d augm_vrng
#python3.6 augment.py -d augm_frng -r

python3.6 video2tfrecord.py -m 1 -s test -c lab -n raw -f single
#python3.6 video2tfrecord.py -m 1 -s val -c rgb -n fnorm -t fs -f single
#python3.6 video2tfrecord.py -m 1 -s val -c rgb -n vnorm -t fs -f single 
#python3.6 video2tfrecord.py -m 1 -s val -c rgb -n vnorm -t ss -f single
#python3.6 video2tfrecord.py -m 1 -s val -c rgb -n fnorm -t ss -f single 

