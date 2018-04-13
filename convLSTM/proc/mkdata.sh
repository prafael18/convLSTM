#!/usr/bin/env bash
./video2tfrecord.py -m 1 -s train -c rgb -n raw -f many -d rgb_raw
./video2tfrecord.py -m 1 -s val -c rgb -n raw -f single
./video2tfrecord.py -m 1 -s test -c rgb -n raw -f single