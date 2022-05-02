import os
import wandb

import argparse
import random
import shutil
import time
import warnings
import torch
import glob
import pathlib
import os.path as path
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
from collections import defaultdict
from yaml import safe_load

from common.datasets import load_cifar, TransformingTensorDataset, get_cifar_data_aug, load_cifar10_1
from common.datasets import load_cifar550, load_svhn_all, load_svhn, load_cifar5m
import common.models32 as models
from common import load_state_dict
from .utils import get_model32, mse_loss, test_all, get_dataset, add_noise, get_data_aug, make_loader

from common.logging import VanillaLogger


parser = argparse.ArgumentParser(description='vanilla testing')
parser.add_argument('--proj', default='test-soft', type=str, help='project name')
parser.add_argument('--dataset', default='cifar5m', type=str, help='dataset model was trained on')
parser.add_argument('--corr', default='', type=str)
parser.add_argument('--eval-id-vs-ood', dest='eval_id_vs_ood', default=False, action='store_true')
parser.add_argument('--eval-calibration-metrics', dest='eval_calibration_metrics', default=False, action='store_true')
parser.add_argument('--resume', default=0, type=int, help='resume at step')
parser.add_argument('--id', default='', type=str, help='wandb id to resume')

parser.add_argument('--nsamps', default=50000, type=int, help='num. train samples')
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--iid', default=False, action='store_true', help='simulate infinite samples (fresh samples each batch)')

parser.add_argument('--arch', metavar='ARCH', default='preresnet18')
parser.add_argument('--pretrained', type=str, default=None, help='expanse path to ensemble model dir')
parser.add_argument('--datadir', type=str, default='~/tmp/data', help='path to data')
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

# dict mapping dataset eval names to generic dataset names
dataset_names = {'base_cifar10_train': 'cifar10', 'base_cifar10_val': 'cifar10'}

# dict mapping dataset eval names to wandb logging names
dataset_logs = {'base_cifar10_train': 'Train', 'base_cifar10_val': 'Test', 'base_cifar10_test': 'CF10', 'cifar10_1': 'CF10.1', 'cifar10c': 'CF10-C', 'cifar5m': 'CF5m'}

default_subset = 'all'

def get_loader(args):
    (X_tr, Y_tr, X_te, Y_te), preproc = get_dataset(args.dataset)
    val_set = TransformingTensorDataset(X_te, Y_te, transform=preproc)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=256,
        shuffle=False, num_workers=args.workers, pin_memory=True)
    return test_loader

def get_step(f):
    final_subdir = pathlib.PurePath(f).parent.name
    if final_subdir.startswith('step'):
        return int(final_subdir.replace('step', ''))
    else:
        return -1

def get_run_id():
    return path.basename(path.normpath(args.pretrained))

def get_wandb_name(args):
    return f'{args.arch}-{args.dataset} n={args.nsamps}'

def main():
    ## argparsing hacks
    if args.pretrained == 'None':
        args.pretrained = None # hack for caliban

    if args.resume:
        id = args.id if args.id else get_run_id()
        wandb.init(project=args.proj, entity='deep-bootstrap2', id=id, resume='allow')
    else:
        wandb.init(project=args.proj, entity='deep-bootstrap2')
        wandb.run.name = wandb.run.id  + " - " + get_wandb_name(args)
    cudnn.benchmark = True

    with open('config.yml', 'r') as fp:
        config = safe_load(fp)

    if args.dataset.startswith('cifar5m-binary'):
        nclasses = 2
    else:
        nclasses = 10

    # init logging
    logger = VanillaLogger(args, wandb, expanse_root=config['expanse_root'], hash=True)

    print('Loading dataset...')
    test_loader = get_loader(args)

    print('Done loading.')

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda() if args.loss == 'xent' else mse_loss


    model_dirs = []
    for f in os.listdir(args.pretrained):
        if os.path.isdir(os.path.join(args.pretrained, f)):
            model_dirs.append(os.path.join(args.pretrained, f))

    print(model_dirs)


    # order models based on step
    steps = []
    for model in model_dirs:
        for f in glob.glob(os.path.join(model, "**/model.pt"), recursive=True):
            step = get_step(f)
            if step != -1:
                steps.append(step)
        steps.sort()
        if len(steps) != 0:
            break
    print(steps)

    for step in steps:
        if step < args.resume:
            continue

        models = []
        for model_dir in model_dirs:


            f = f'{model_dir}/step{step:06}/model.pt'

            if os.path.exists(f):
                #load the model
                model = get_model32(args, args.arch, half=args.half, nclasses=nclasses, pretrained_path=f)
                if args.pretrained:
                    load_state_dict(model, f)

                # model = torch.nn.DataParallel(model).cuda()
                if torch.cuda.is_available():
                    model.cuda()

                models.append(model)
                print(f'Evaluating model {get_wandb_name(args)} at step {step}')

        d = {}

        fname = f'{args.datadir}calibration/{wandb.run.id}_{step}.pickle'
        d.update({'batch_num' : step})
        results = test_all(test_loader, models, criterion, half=args.half, calibration_metrics=args.eval_calibration_metrics, fname=fname)
        for k, v in results.items():
            print(f'{args.dataset} {k} : {v}')
            d.update({ f'{args.dataset} {k}' : v for k, v in results.items()})

        if step != -1:
            logger.log_scalars(d, step=step)
        else:
            logger.wandb.summary.update(d)

if __name__ == '__main__':
    main()
