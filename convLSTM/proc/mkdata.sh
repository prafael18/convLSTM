#!/usr/bin/env bash

# ./video2tfrecord.py -m 3 -s train -c rgb -n raw -f many -l fs -d raw_rgb_lfs

./video2tfrecord.py -m 3 -s train -c rgb -n raw -f single

#./video2tfrecord.py -m 3 -s test -c rgb -n fnorm -t fs -f single
#./video2tfrecord.py -m 3 -s test -c lab -n vnorm -t ss -f single
#./video2tfrecord.py -m 3 -s train -c lab -n raw -f many -d raw_lab
#./video2tfrecord.py -m 3 -s val -c rgb -n raw -f single
#./video2tfrecord.py -m 3 -s val -c lab -n raw -f single
