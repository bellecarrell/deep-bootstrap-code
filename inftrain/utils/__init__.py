#from google.cloud import storage
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import subprocess
import os
import time
import shutil
from datetime import datetime
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from common.datasets import load_cifar, TransformingTensorDataset, get_cifar_data_aug, load_cifar500, load_cifar10_1, load_cifar5m, load_cifar5m_test
import common.models32 as models


def get_optimizer(optimizer_name, parameters, lr, momentum=0, weight_decay=0):
    if optimizer_name == 'sgd':
        return optim.SGD(parameters, lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'nesterov_sgd':
        return optim.SGD(parameters, lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def get_scheduler(args, scheduler_name, optimizer, num_epochs, batches_per_epoch, **kwargs):
    if args.opt == 'adam':
        # constant LR sched
        print("Using adam with const LR.")
        return lr_scheduler.StepLR(optimizer, num_epochs, gamma=1, **kwargs)
    if args.sched is not None:
        sched = list(map(int, args.sched.split(',')))
        print("Using step-wise LR schedule:", sched)
        return lr_scheduler.MultiStepLR(optimizer, milestones=np.cumsum(sched[:-1]), gamma=0.1)

    if scheduler_name == 'const':
        return lr_scheduler.StepLR(optimizer, num_epochs, gamma=1, **kwargs)
    elif scheduler_name == '3step':
        return lr_scheduler.StepLR(optimizer, round(num_epochs / 3), gamma=0.1, **kwargs)
    elif scheduler_name == 'exponential':
        return lr_scheduler.ExponentialLR(optimizer, (1e-3) ** (1 / num_epochs), **kwargs)
    elif scheduler_name == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, **kwargs)
    elif scheduler_name == 'onecycle':
        return lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=batches_per_epoch, epochs=num_epochs)

def get_rrand_model(args, nclasses=10):
    ''' Returns random model with random width and num_cells. '''
    from nas import get_random_model
    import sys
    import random
    if args.aseed is None:
        args.aseed = random.randrange(sys.maxsize)

    r = random.Random(args.aseed+1)
    width = r.choice([16, 64])
    num_cells = r.choice([1, 5])
    model, genotype = get_random_model(seed=args.aseed, num_classes=nclasses, width=width, num_cells=num_cells, max_nodes=4)
    print('genotype: ', genotype.tostr())
    print(f'width={width}, num_cells={num_cells}')
    args.genotype = genotype.tostr()
    args.width = width
    args.num_cells = num_cells
    return model

def get_model32(args, model_name, nchannels=3, nclasses=10, half=False, pretrained_path=None):
    ngpus = torch.cuda.device_count()
    print("=> creating model '{}'".format(model_name))
    if model_name.startswith('mlp'): # eg: mlp[512,512,512]
        widths = eval(model_name[3:])
        model = models.mlp(widths=widths, num_classes=nclasses, pretrained_path=pretrained_path)
    elif model_name == 'rand5':
        from nas import get_random_model
        num_cells = 5
        seed = args.aseed if 'aseed' in args else None
        model, genotype = get_random_model(seed=seed, num_classes=nclasses, width=64, num_cells=num_cells, max_nodes=4)
        print('genotype: ', genotype.tostr())
        args.genotype = genotype.tostr()
    elif model_name == 'rrand':
        model = get_rrand_model(args, nclasses=nclasses)
    elif model_name.startswith('vit'):
        model = models.__dict__[model_name](num_classes=nclasses, pretrained_path=pretrained_path)
    else:
        if args.width is not None:
            model = models.__dict__[model_name](num_classes=nclasses, width=args.width)
        else:
            model = models.__dict__[model_name](num_classes=nclasses)

    args.nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num model parameters:", args.nparams)
    if half:
        print('Using half precision except in Batch Normalization!')
        model = model.half()
        for module in model.modules():
            if (isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d)):
                module.float()
    return model




class AverageMeter(object):
    def __init__(self, name=None):
        self.reset()
        self.name=name

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
        return f'[{self.name}]:{self.avg}'


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


def recycle(iterable):
    """Variant of itertools.cycle that does not save iterates."""
    while True:
        for i in iterable:
            yield i



def cuda_transfer(images, target, half=False):
    images = images.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    if half: images = images.half()
    return images, target

def mse_loss(output, y):
    y_true = F.one_hot(y, 10).float()
    return (output - y_true).pow(2).sum(-1).mean()

def predict(loader, model):
    # switch to evaluate mode
    model.eval()
    n = 0
    predsAll = []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = cuda_transfer(images, target)
            output = model(images)

            preds = output.argmax(1).long().cpu()
            predsAll.append(preds)

    preds = torch.cat(predsAll)
    return preds

def expected_calibration_error(y_true, y_pred, num_bins=10):
  pred_y = np.argmax(y_pred, axis=-1)
  correct = (pred_y == y_true).astype(np.float32)
  prob_y = np.max(y_pred, axis=-1)

  b = np.linspace(start=0, stop=1.0, num=num_bins)
  bins = np.digitize(prob_y, bins=b, right=True)

  o = 0
  for b in range(num_bins):
    mask = bins == b
    if np.any(mask):
        o += np.abs(np.sum(correct[mask] - prob_y[mask]))

  return o / y_pred.shape[0]

def static_calibration_error(y_true, y_pred, num_bins=10):
  classes = y_pred.shape[-1]

  o = 0
  for cur_class in range(classes):
      correct = (cur_class == y_true).astype(np.float32)
      prob_y = y_pred[..., cur_class]

      b = np.linspace(start=0, stop=1.0, num=num_bins)
      bins = np.digitize(prob_y, bins=b, right=True)

      for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

  return o / (y_pred.shape[0] * classes)

def test_all(loader, model, criterion, calibration_metrics=False):
    # switch to evaluate mode
    model.eval()
    aloss = AverageMeter('Loss')
    aerr = AverageMeter('Error')
    asoft = AverageMeter('SoftError')
    mets = [aloss, aerr, asoft]

    if calibration_metrics:
        y_pred = []
        y_true = []

    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            bs = len(images)
            if torch.cuda.is_available():
                images, target = cuda_transfer(images, target, half=half)
            output = model(images)
            loss = criterion(output, target)

            err = (output.argmax(1) != target).float().mean().item()
            p = F.softmax(output, dim=1) # [bs x 10] : softmax probabilties
            p_corr = p.gather(1, target.unsqueeze(1)).squeeze() # [bs]: prob on the correct label
            soft = (1-p_corr).mean().item()

            aloss.update(loss.item(), bs)
            aerr.update(err, bs)
            asoft.update(soft, bs)

            if calibration_metrics:
                y_pred.append(p)
                y_true.append(target)

    results = {m.name : m.avg for m in mets}
    if calibration_metrics:
        y_pred = torch.cat(y_pred).numpy()
        y_true = torch.cat(y_true).numpy()
        results.update({'ece': expected_calibration_error(y_true, y_pred), 'sce': static_calibration_error(y_true, y_pred)})

    return results

def get_dataset(dataset, test_only=False):
    '''
        Returns dataset and pre-transform (to process dataset into [-1, 1] torch tensor)
    '''

    noop = transforms.Compose([])
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    uint_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize]) # numpy unit8 --> [-1, 1] tensor

    if dataset == 'cifar10':
        return load_cifar(), noop
    if dataset == 'cifar550':
        return load_cifar550(), noop
    if dataset == 'cifar5m':
        if test_only:
            return load_cifar5m_test(), uint_transform
        return load_cifar5m(), uint_transform


def add_noise(Y, p: float):
    ''' Adds noise to Y, s.t. the label is wrong w.p. p '''
    num_classes = torch.max(Y).item()+1
    print('num classes: ', num_classes)
    noise_dist = torch.distributions.categorical.Categorical(
        probs=torch.tensor([1.0 - p] + [p / (num_classes-1)] * (num_classes-1)))
    return (Y + noise_dist.sample(Y.shape)) % num_classes


def get_data_aug(aug : int):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    unnormalize = transforms.Compose([
        transforms.Normalize((0, 0, 0), (2, 2, 2)),
        transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1))
    ])

    if aug == 0:
        print('data-aug: NONE')
        return transforms.Compose([])
    elif aug == 1:
        print('data-aug: flips only')
        return transforms.Compose(
            [unnormalize,
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
    elif aug == 2:
        print('data-aug: full')
        return transforms.Compose(
            [unnormalize,
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
    elif aug == 3:
        print('data-aug: full (reflect-crop)')
        return transforms.Compose(
            [unnormalize,
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])


def make_loader(x, y, transform=None, batch_size=256, num_workers=1):
    dataset = TransformingTensorDataset(x, y, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader

def make_loader_cifar10_1(args):
    datadir = '~/tmp/data/'
    if args.datadir:
        datadir = args.datadir
    data, targets = load_cifar10_1('v4', datadir=datadir)
    preprocess = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)])
    test_transform = preprocess
    cifar10_1 = datasets.CIFAR10(datadir, train=False, transform=test_transform, download=True)
    cifar10_1.data = data
    cifar10_1.targets = torch.tensor(targets, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
            cifar10_1,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True)
    return loader

def get_wandb_name(args):
    return f'{args.arch}-{args.dataset} n={args.nsamps} aug={args.aug} iid={args.iid}'
