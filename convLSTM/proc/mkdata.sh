#!/usr/bin/env bash
./video2tfrecord.py -m 1 -s val -t fs -n vnorm -c rgb -f single
./video2tfrecord.py -m 1 -s test -t fs -n vnorm -c rgb -f single
./video2tfrecord.py -m 1 -s train -t fs -n vnorm -c rgb -f many -d rgb_fs_vnorm
./video2tfrecord.py -m 1 -s val -t fs -n vnorm -c lab -f single
./video2tfrecord.py -m 1 -s val -t fs -n vnorm -c lab -f single
./video2tfrecord.py -m 1 -s train -t fs -n vnorm -c lab -f many -d lab_fs_vnorm
./video2tfrecord.py -m 1 -s val -t fs -n fnorm -c rgb -f single
./video2tfrecord.py -m 1 -s test -t fs -n fnorm -c rgb -f single
./video2tfrecord.py -m 1 -s train -t fs -n fnorm -c rgb -f many -d rgb_fs_fnorm
./video2tfrecord.py -m 1 -s val -t fs -n fnorm -c lab -f single
./video2tfrecord.py -m 1 -s val -t fs -n fnorm -c lab -f single
./video2tfrecord.py -m 1 -s train -t fs -n fnorm -c lab -f many -d lab_fs_fnorm


#cp -r /home/panda/ic/data/train/lab_ss_vnorm /home/storage_local/panda/data/train
#cp /home/panda/ic/data/val/val_ss_vnorm_lab.tfrecords /home/storage_local/panda/data/val
#python3 /home/panda/ic/convLSTM/train.py 2>&1 | tee -a /home/panda/ic/convLSTM/stdout.txt
