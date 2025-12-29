python IGD/train_ckpts.py -d nette --imagenet_dir /data2/wlf/datasets/imagenet/train/ /data2/wlf/datasets/imagenet \
    -n convnet6 --nclass 10 --norm_type instance --tag test --slct_type random --spec nette --mixup vanilla --repeat 1 \
    --ckpt-dir ./ckpts/nette/convnet6/ --epochs 50 --depth 6 --ipc -1
