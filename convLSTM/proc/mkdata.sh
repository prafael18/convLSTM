#!/usr/bin/env bash
./video2tfrecord.py -m 1 -s train -c lab -n raw -f many -d lab_raw
./video2tfrecord.py -m 1 -s val -c lab -n raw -f single
./video2tfrecord.py -m 1 -s test -c lab -n raw -f single
python3 /home/panda/ic/convLSTM/train.py 2>&1 | tee -a /home/panda/ic/convLSTM/stdout.txt
