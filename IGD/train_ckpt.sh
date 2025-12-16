#!/bin/bash
model=convnet6
spec=1k

python train_ckpts.py -d imagenet --imagenet_dir /data2/wlf/datasets/imagenet/train/ /data2/wlf/datasets/imagenet/val/ \
    -n ${model} --nclass 1000 --norm_type instance --tag test --slct_type random --spec ${spec} --mixup vanilla --repeat 1 \
    --ckpt-dir ./ckpts/${spec}/${model}/ --epochs 50 --depth 6 --ipc -1
