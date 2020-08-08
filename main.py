import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import shutil
import time
import torchvision.models as models

from config import get_args
from data.dataset import load_data
from loss.focal_loss import FocalLoss
from model.ghostnet import ghostnet
from model.mobilenetv3 import mobilenetv3_small, mobilenetv3_large
from model.model import mixnet_s, mixnet_m, mixnet_l
from tensorboardX import SummaryWriter


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# reference,
# https://github.com/pytorch/examples/blob/master/imagenet/main.py
# Thank you.
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(model, train_loader, optimizer, criterion, epoch, args, logger, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1, top5,
                             prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            data = data.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))
        n_iter = (epoch - 1) * len(train_loader) + i + 1
        writer.add_scalar('Train/Loss', loss.item(), n_iter)
        writer.add_scalar('Train/Acc1', acc1[0], n_iter)
        writer.add_scalar('Train/Acc5', acc5[0], n_iter)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_interval == 0:
            progress.print(i)


def eval(model, val_loader, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            if args.cuda is not None:
                data = data.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_interval == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def main(args, logger):
    writer = SummaryWriter(log_dir=os.path.join('logs', args.dataset, args.model_name, args.loss))

    train_loader, test_loader = load_data(args)
    if args.dataset == 'CIFAR10':
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        num_classes = 100
    elif args.dataset == 'TINY_IMAGENET':
        num_classes = 200
    elif args.dataset == 'IMAGENET':
        num_classes = 1000

    print('Model name :: {}, Dataset :: {}, Num classes :: {}'.format(args.model_name, args.dataset, num_classes))
    if args.model_name == 'mixnet_s':
        model = mixnet_s(num_classes=num_classes, dataset=args.dataset)
        # model = mixnet_s(num_classes=num_classes)
    elif args.model_name == 'mixnet_m':
        model = mixnet_m(num_classes=num_classes, dataset=args.dataset)
    elif args.model_name == 'mixnet_l':
        model = mixnet_l(num_classes=num_classes, dataset=args.dataset)
    elif args.model_name == 'ghostnet':
        model = ghostnet(num_classes=num_classes)
    elif args.model_name == 'mobilenetv2':
        model = models.mobilenet_v2(num_classes=num_classes)
    elif args.model_name == 'mobilenetv3_s':
        model = mobilenetv3_small(num_classes=num_classes)
    elif args.model_name == 'mobilenetv3_l':
        model = mobilenetv3_large(num_classes=num_classes)
    else:
        raise NotImplementedError

    if args.pretrained_model:
        filename = 'best_model_' + str(args.dataset) + '_' + str(args.model_name) + '_ckpt.tar'
        print('filename :: ', filename)
        file_path = os.path.join('./checkpoint', filename)
        checkpoint = torch.load(file_path)

        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        best_acc5 = checkpoint['best_acc5']
        model_parameters = checkpoint['parameters']
        print('Load model, Parameters: {0}, Start_epoch: {1}, Acc1: {2}, Acc5: {3}'.format(model_parameters, start_epoch, best_acc1, best_acc5))
        logger.info('Load model, Parameters: {0}, Start_epoch: {1}, Acc1: {2}, Acc5: {3}'.format(model_parameters, start_epoch, best_acc1, best_acc5))
    else:
        start_epoch = 1
        best_acc1 = 0.0
        best_acc5 = 0.0

    if args.cuda:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.cuda()

    print("Number of model parameters: ", get_model_parameters(model))
    logger.info("Number of model parameters: {0}".format(get_model_parameters(model)))

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'focal':
        criterion = FocalLoss()
    else:
        raise NotImplementedError
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.001)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100], gamma=0.2) #learning rate decay

    for epoch in range(start_epoch, args.epochs + 1):
        # adjust_learning_rate(optimizer, epoch, args)
        train(model, train_loader, optimizer, criterion, epoch, args, logger, writer)
        acc1, acc5 = eval(model, test_loader, criterion, args)
        lr_scheduler.step()

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            best_acc5 = acc5

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        filename = 'model_' + str(args.dataset) + '_' + str(args.model_name) + '_ckpt.tar'
        print('filename :: ', filename)

        parameters = get_model_parameters(model)

        if torch.cuda.device_count() > 1:
            save_checkpoint({
                'epoch': epoch,
                'arch': args.model_name,
                'state_dict': model.module.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
                'optimizer': optimizer.state_dict(),
                'parameters': parameters,
            }, is_best, filename)
        else:
            save_checkpoint({
                'epoch': epoch,
                'arch': args.model_name,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
                'optimizer': optimizer.state_dict(),
                'parameters': parameters,
            }, is_best, filename)
        writer.add_scalar('Test/Acc1', acc1, epoch)
        writer.add_scalar('Test/Acc5', acc5, epoch)

        print(" Test best acc1:", best_acc1, " acc1: ", acc1, " acc5: ", acc5)
    writer.close()


def save_checkpoint(state, is_best, filename):
    file_path = os.path.join('./checkpoint', filename)
    torch.save(state, file_path)
    best_file_path = os.path.join('./checkpoint', 'best_' + filename)
    if is_best:
        print('best Model Saving ...')
        shutil.copyfile(file_path, best_file_path)


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args, logger = get_args()
    set_random_seed(0, True)
    main(args, logger)
