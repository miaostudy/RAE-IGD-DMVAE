#!/bin/bash

spec=woof
net=resnet_ap # resnet_ap resnet
d=10
ipc=10
path=./results/dit-distillation/woof-10-rae-igd-ckpts-resnet_ap-k5-gamma50-r1-gi200-low30-high45

python IGD/train.py -d imagenet --imagenet_dir ${path} /data/wlf/datasets/imagenet/ -n ${net} --depth ${d} --nclass 10 --norm_type instance --ipc ${ipc} --tag test --slct_type random --spec ${spec} --batch_size 16


