#!/bin/bash

spec=nette
net=resnet # resnet_ap resnet
d=18
ipc=10
path=./results/igd_time_window

python IGD/train.py -d imagenet --imagenet_dir ${path} /data/wlf/datasets/imagenet/ -n ${net} --depth ${d} --nclass 10 --norm_type instance --ipc ${ipc} --tag test --slct_type random --spec ${spec} --batch_size 16


