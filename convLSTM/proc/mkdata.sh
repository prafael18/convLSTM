#!/usr/bin/env bash
./video2tfrecord.py -m 0 -s test -c rgb -t raw -f single
./video2tfrecord.py -m 0 -s test -c lab -t ss -n vnorm -f single
#./video2tfrecord.py -m 1 -s train -c lab -t ss -n vnorm -f many -d lab_ss_vnorm
#cp -r /home/panda/ic/data/train/lab_ss_vnorm /home/storage_local/panda/data/train
#cp /home/panda/ic/data/val/val_ss_vnorm_lab.tfrecords /home/storage_local/panda/data/val
#python3 /home/panda/ic/convLSTM/train.py 2>&1 | tee -a /home/panda/ic/convLSTM/stdout.txt
