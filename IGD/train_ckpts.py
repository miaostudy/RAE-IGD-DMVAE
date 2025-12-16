# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as models
import train_models.resnet as RN
import train_models.resnet_ap as RNAP
import train_models.convnet as CN
import train_models.densenet_cifar as DN
from data import load_data, MEANS, STDS
from misc.utils import random_indices, rand_bbox, AverageMeter, accuracy, get_time, Plotter
from efficientnet_pytorch import EfficientNet
import warnings
from tqdm import tqdm, trange  # 新增：导入tqdm进度条库

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
warnings.filterwarnings("ignore")

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

mean_torch = {}
std_torch = {}
for key, val in MEANS.items():
    mean_torch[key] = torch.tensor(val, device=device).reshape(1, len(val), 1, 1)
for key, val in STDS.items():
    std_torch[key] = torch.tensor(val, device=device).reshape(1, len(val), 1, 1)


def define_model(args, nclass, logger=None, size=None):
    """Define neural network models
    """
    if size is None:
        size = args.size

    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset,
                          args.depth,
                          nclass,
                          norm_type=args.norm_type,
                          size=size,
                          nch=args.nch)
    elif args.net_type == 'resnet_ap':
        model = RNAP.ResNetAP(args.dataset,
                              args.depth,
                              nclass,
                              width=args.width,
                              norm_type=args.norm_type,
                              size=size,
                              nch=args.nch)
    elif args.net_type == 'efficient':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=nclass)
    elif args.net_type == 'densenet':
        model = DN.densenet_cifar(nclass)
    elif args.net_type == 'convnet':
        model = CN.ConvNet(channel=args.nch, num_classes=nclass, net_width=128, net_depth=3,
                           net_act='relu', net_norm='instancenorm', net_pooling='avgpooling',
                           im_size=(args.size, args.size))
    elif args.net_type == 'convnet6':
        model = CN.ConvNet(channel=args.nch, num_classes=nclass, net_width=128, net_depth=6,
                           net_act='relu', net_norm='instancenorm', net_pooling='avgpooling',
                           im_size=(args.size, args.size))
    elif args.net_type == 'convnet4':
        model = CN.ConvNet(channel=args.nch, num_classes=nclass, net_width=128, net_depth=4,
                           net_act='relu', net_norm='instancenorm', net_pooling='avgpooling',
                           im_size=(args.size, args.size))
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    if logger is not None:
        logger(f"=> creating model {args.net_type}-{args.depth}, norm: {args.norm_type}")

    return model


def main(args, logger, repeat=1):
    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True
    logger(f"ImageNet directory: {args.imagenet_dir[0]}")
    print('spec:', args.spec)
    _, train_loader, val_loader, nclass = load_data(args)

    best_acc_l = []
    acc_l = []
    global trajectories
    trajectories = []

    for i in tqdm(range(repeat), desc="Total Training Repeats", leave=True, unit="repeat"):
        logger(f"Repeat: {i + 1}/{repeat}")
        plotter = Plotter(args.save_dir, args.epochs, idx=i)

        model = define_model(args, nclass, logger)
        tqdm.write('Define model completed')  # 不覆盖进度条的打印

        best_acc, acc, timestamps = train(args, model, train_loader, val_loader, plotter, logger)
        trajectories.append(timestamps)
        best_acc_l.append(best_acc)
        acc_l.append(acc)

        if len(trajectories) == args.save_interval:
            n = int(args.start)
            while os.path.exists(os.path.join(args.ckpt_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            save_path = os.path.join(args.ckpt_dir, "replay_buffer_{}.pt".format(n))
            tqdm.write(f"Saving trajectory to {save_path}")
            torch.save(trajectories, save_path)
            trajectories = []

    avg_best = np.mean(best_acc_l)
    std_best = np.std(best_acc_l)
    avg_last = np.mean(acc_l)
    std_last = np.std(acc_l)
    print(
        f'\n(expert {repeat}) Best acc: {avg_best:.1f} ± {std_best:.1f} | Last acc: {avg_last:.1f} ± {std_last:.1f}')


def train(args, model, train_loader, val_loader, plotter=None, logger=None):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(),
                          args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    cur_epoch, best_acc1, best_acc5, acc1, acc5 = 0, 0, 0, 0, 0
    if args.pretrained:
        pretrained = "{}/{}".format(args.save_dir, 'checkpoint.pth.tar')
        cur_epoch, best_acc1 = load_checkpoint(pretrained, model, optimizer)

    model = model.to(device)
    logger(f"Start training with base augmentation and {args.mixup} mixup")

    timestamps = []
    timestamps.append([p.detach().cpu() for p in model.parameters()])
    args.epoch_print_freq = 1

    epoch_range = trange(cur_epoch + 1, args.epochs + 1, desc="Training Epochs", leave=True, unit="epoch")
    for epoch in epoch_range:
        acc1_tr, top5_tr, loss_tr = train_epoch(args,
                                                train_loader,
                                                model,
                                                criterion,
                                                optimizer,
                                                epoch,
                                                logger,
                                                mixup=args.mixup)

        timestamps.append([p.detach().cpu() for p in model.parameters()])

        if epoch % args.epoch_print_freq == 0:
            acc1, acc5, loss_val = validate(args, val_loader, model, criterion, epoch, logger)
            tqdm.write(f'Epoch {epoch} | Val Top1: {acc1:.2f}% | Val Loss: {loss_val:.4f}')

            if plotter is not None:
                plotter.update(epoch, acc1_tr, acc1, loss_tr, loss_val)

            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_acc5 = acc5
                if logger is not None and args.verbose == True:
                    logger(f'Best accuracy (top-1/5): {best_acc1:.1f} / {best_acc5:.1f}')

            epoch_range.set_postfix({
                'Best Val Top1': f'{best_acc1:.2f}%',
                'Cur Val Top1': f'{acc1:.2f}%',
                'Train Loss': f'{loss_tr:.4f}',
                'Val Loss': f'{loss_val:.4f}'
            })

    return best_acc1, acc1, timestamps


def train_epoch(args,
                train_loader,
                model,
                criterion,
                optimizer,
                epoch=0,
                logger=None,
                mixup='vanilla',
                n_data=-1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    num_exp = 0

    train_pbar = tqdm(enumerate(train_loader),
                      desc=f"Epoch {epoch} Train",
                      total=len(train_loader),
                      leave=False,
                      unit="batch")

    for i, (input, target) in train_pbar:
        if train_loader.device == 'cpu':
            input = input.to(device)
            target = target.to(device)

        data_time.update(time.time() - end)

        r = np.random.rand(1)
        if r < args.mix_p and mixup == 'cut':
            lam = np.random.beta(args.beta, args.beta)
            rand_index = random_indices(target, nclass=args.nclass)
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            output = model(input)
            loss = criterion(output, target) * ratio + criterion(output, target_b) * (1. - ratio)
        else:
            output = model(input)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        train_pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Top1': f'{top1.avg:.2f}%',
            'Top5': f'{top5.avg:.2f}%',
            'DataTime': f'{data_time.avg:.2f}s',
            'BatchTime': f'{batch_time.avg:.2f}s'
        })

        num_exp += len(target)
        if (n_data > 0) and (num_exp >= n_data):
            train_pbar.close()
            break

    if (epoch % args.epoch_print_freq == 0) and (logger is not None) and args.verbose == True:
        logger(
            '(Train) [Epoch {0}/{1}] {2} Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}'
            .format(epoch, args.epochs, get_time(), top1=top1, top5=top5, loss=losses))

    return top1.avg, top5.avg, losses.avg


def validate(args, val_loader, model, criterion, epoch, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()

    val_pbar = tqdm(enumerate(val_loader),
                    desc=f"🔍 Epoch {epoch} Val",
                    total=len(val_loader),
                    leave=False,
                    unit="batch")

    with torch.no_grad():
        for i, (input, target) in val_pbar:
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            val_pbar.set_postfix({
                'Val Loss': f'{losses.avg:.4f}',
                'Val Top1': f'{top1.avg:.2f}%',
                'Val Top5': f'{top5.avg:.2f}%',
                'BatchTime': f'{batch_time.avg:.2f}s'
            })

    val_pbar.close()

    # 打印验证日志
    if logger is not None and args.verbose == True:
        logger(
            '(Test ) [Epoch {0}/{1}] {2} Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}'
            .format(epoch, args.epochs, get_time(), top1=top1, top5=top5, loss=losses))

    return top1.avg, top5.avg, losses.avg


def load_checkpoint(path, model, optimizer):
    if os.path.isfile(path):
        tqdm.write(f"=> loading checkpoint '{path}'")
        checkpoint = torch.load(path, map_location=device)  # 指定设备，避免cuda:0
        checkpoint['state_dict'] = dict(
            (key[7:], value) for (key, value) in checkpoint['state_dict'].items())
        model.load_state_dict(checkpoint['state_dict'])
        cur_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        tqdm.write(f"=> loaded checkpoint '{path}' (epoch: {cur_epoch}, best acc1: {best_acc1}%)")
    else:
        tqdm.write(f"=> no checkpoint found at '{path}'")
        cur_epoch = 0
        best_acc1 = 100

    return cur_epoch, best_acc1


def save_checkpoint(save_dir, state, is_best):
    os.makedirs(save_dir, exist_ok=True)
    if is_best:
        ckpt_path = os.path.join(save_dir, 'model_best.pth.tar')
    else:
        ckpt_path = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, ckpt_path)
    tqdm.write(f"Checkpoint saved! {ckpt_path}")


if __name__ == '__main__':
    from misc.utils import Logger
    from argument import args

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")

    main(args, logger, args.repeat)