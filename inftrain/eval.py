import os
import wandb

import argparse
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from tqdm.auto import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset
import torch.optim as optim
from torch.optim import lr_scheduler

from common.datasets import load_cifar, TransformingTensorDataset, get_cifar_data_aug
from common.datasets import load_cifar550, load_svhn_all, load_svhn, load_cifar5m
import common.models32 as models
from common import load_state_dict
from .utils import get_model32, mse_loss, test_all, get_dataset, add_noise, get_data_aug, make_loader

from common.logging import VanillaLogger


parser = argparse.ArgumentParser(description='vanilla testing')
parser.add_argument('--proj', default='test-soft', type=str, help='project name')
parser.add_argument('--dataset', default='cifar5m', type=str, choices=['base_cifar10_train', 'base_cifar10_val', 'base_cifar10_test'])
parser.add_argument('--nsamps', default=50000, type=int, help='num. train samples')
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--iid', default=False, action='store_true', help='simulate infinite samples (fresh samples each batch)')

parser.add_argument('--arch', metavar='ARCH', default='preresnet18')
parser.add_argument('--pretrained', type=str, default=None, help='expanse path to pretrained model state dict (optional)')
parser.add_argument('--width', default=None, type=int, help="architecture width parameter (optional)")
parser.add_argument('--loss', default='xent', choices=['xent', 'mse'], type=str)
parser.add_argument('--aug', default=0, type=int, help='data-aug (0: none, 1: flips, 2: all)')

# for keeping the same LR sched across different samp sizes.
parser.add_argument('--nbatches', default=None, type=int, help='Total num. batches to train for. If specified, overrides EPOCHS.')

parser.add_argument('--noise', default=0.0, type=float, help='label noise probability (train & test).')

parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--half', default=False, action='store_true', help='training with half precision')
parser.add_argument('--comment', default=None)

args = parser.parse_args()

# dict mapping dataset eval names to generic dataset names used elsewhere
dataset_names = {'base_cifar10_train': 'cifar10', 'base_cifar10_val': 'cifar10'}

def get_loader():
    if args.dataset.startswith('base_cifar10') and not args.dataset.endswith('test'):
        (X_tr, Y_tr, X_te, Y_te), preproc = get_dataset(dataset_names[args.dataset])

        # subsample
        if not args.iid:
            I = np.random.permutation(len(X_tr))[:args.nsamps]
            X_tr, Y_tr = X_tr[I], Y_tr[I]

        # Add noise (optionally)
        Y_tr = add_noise(Y_tr, args.noise)
        Y_te = add_noise(Y_te, args.noise)

        tr_set = TransformingTensorDataset(X_tr, Y_tr, transform=transforms.Compose([preproc, get_data_aug(args.aug)]))
        val_set = TransformingTensorDataset(X_te, Y_te, transform=preproc)

        tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=args.batchsize,
                shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True) # drop the last batch if it's incomplete (< batch size)
        te_loader = torch.utils.data.DataLoader(val_set, batch_size=256,
                shuffle=False, num_workers=args.workers, pin_memory=True)

        if args.dataset.endswith('train'):
            return tr_loader
        elif args.dataset.endswith('val'):
            return te_loader
    elif args.dataset == 'base_cifar10_test':
        return make_loader(*(load_cifar()[2:]))

def main():
    ## argparsing hacks
    if args.pretrained == 'None':
        args.pretrained = None # hack for caliban

    wandb.init(project=args.proj)
    cudnn.benchmark = True

    #load the model
    model = get_model32(args, args.arch, half=args.half, nclasses=10, pretrained_path=args.pretrained)
    if args.pretrained:
        load_state_dict(model, args.pretrained)

    # model = torch.nn.DataParallel(model).cuda()
    if torch.cuda.is_available():
        model.cuda()

    # init logging
    logger = VanillaLogger(args, wandb, hash=True)

    print('Loading dataset...')
    test_loader = get_loader()
    print('Done loading.')

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda() if args.loss == 'xent' else mse_loss

    summary = {}
    summary.update({ f'Final Test on dataset {args.dataset} {k}' : v for k, v in test_all(test_loader, model, criterion).items()})

    logger.log_summary(summary)
    logger.flush()


if __name__ == '__main__':
    main()
