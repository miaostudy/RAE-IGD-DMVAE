#!/bin/bash

k=5
low=30
high=45
gi=200
cp=ckpts
r=1
spec=woof
nsample=10
ntype=resnet_ap
gamma=50
phase=0 # end 7
nclass=10
tart_ncls=10
d=10
model_type=rae

python IGD/sample.py --model DiT-XL/2 --image-size 256 --ckpt pretrained_models/DiT-XL-2-256x256.pt \
    --save-dir "./results/dit-distillation/${spec}-${nsample}-rae-igd-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high}" --data-path /data/wlf/datasets/imagenet/train/ \
    --spec ${spec} --gm-scale ${k} --grad-ipc ${gi} --net-type ${ntype} --depth ${d} \
    --low ${low} --high ${high} --ckpt_path ${cp} --repeat ${r} --num-samples ${nsample} --dev-scale ${gamma} \
    --nclass ${nclass} --phase ${phase} --target_nclass ${tart_ncls} --model_type ${model_type}
