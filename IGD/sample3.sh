#!/bin/bash

num_samples=10
k=1
gamma=1
bs=1
python IGD/sample3.py \
  --config RAE/configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml \
  --dataset imagenet \
  --imagenet_dir /data/wlf/datasets/imagenette \
  --spec nette \
  --gamma ${gamma} \
  --k ${k} \
  --num-sampling_steps 50 \
  --high 400 \
  --low 100 \
  --save-dir ./results/${num_samples}_igd_time_window/k${k}_gamma${gamma}_bs${10} \
  --num-samples ${num_samples} \
  --boundary_scale ${bs}


