#!/usr/bin/env bash

# ./video2tfrecord.py -m 3 -s test -c rgb -n raw -f single
# ./video2tfrecord.py -m 3 -s test -c lab -n raw -f single

./video2tfrecord.py -m 3 -s train -c rgb -n raw -f many -d raw_rgb
./video2tfrecord.py -m 3 -s train -c lab -n raw -f many -d raw_lab
./video2tfrecord.py -m 3 -s val -c rgb -n raw -f single
./video2tfrecord.py -m 3 -s val -c lab -n raw -f single
