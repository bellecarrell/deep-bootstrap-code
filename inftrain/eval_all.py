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
parser.add_argument('--eval-dataset', default='base_cifar10_train', type=str, choices=['base_cifar10_train', 'base_cifar10_val', 'base_cifar10_test', 'cifar10c', 'cifar10_1', 'cifar5m', 'cifar5m-binary-easy', 'cifar5m-binary-hard'])
parser.add_argument('--corr', default='', type=str)
parser.add_argument('--eval-id-vs-ood', dest='eval_id_vs_ood', default=False, action='store_true')
parser.add_argument('--eval-calibration-metrics', dest='eval_calibration_metrics', default=False, action='store_true')
parser.add_argument('--resume', default=0, type=int, help='resume at step')
parser.add_argument('--id', default='', type=str, help='wandb id to resume')

parser.add_argument('--nsamps', default=50000, type=int, help='num. train samples')
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--iid', default=False, action='store_true', help='simulate infinite samples (fresh samples each batch)')

parser.add_argument('--arch', metavar='ARCH', default='preresnet18')
parser.add_argument('--pretrained', type=str, default=None, help='expanse path to pretrained model dir')
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
dataset_logs = {'base_cifar10_train': 'Train', 'base_cifar10_val': 'Test', 'base_cifar10_test': 'CF10', 'cifar10_1': 'CF10.1', 'cifar10c': 'CF10-C', 'cifar5m': 'CF5m', 'cifar5m-binary-hard': 'CF5m-binary'}

corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

default_subset = 'all'

def get_loaders():
    if args.eval_dataset.startswith('base_cifar10') and not args.eval_dataset.endswith('test'):
        (X_tr, Y_tr, X_te, Y_te), preproc = get_dataset(dataset_names[args.eval_dataset])

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

        if args.eval_dataset.endswith('train'):
            return {default_subset: tr_loader}
        elif args.eval_dataset.endswith('val'):
            return {default_subset: te_loader}
    elif args.eval_dataset == 'base_cifar10_test':
        return {default_subset: make_loader(*(load_cifar()[2:]))}
    elif args.eval_dataset == 'cifar10c':
        preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize([0.5] * 3, [0.5] * 3)])
        test_transform = preprocess
        test_loaders = {}
        for corruption in corruptions:
            if args.corr and args.corr != corruption:
                continue
            # Reference to original data is mutated
            test_data = datasets.CIFAR10(args.datadir, train=False, transform=test_transform, download=True)
            base_path = os.path.expanduser(args.datadir) + '/cifar/CIFAR-10-C/'
            test_data.data = np.load(base_path + corruption + '.npy')
            test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=args.batchsize,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True)
            test_loaders[corruption] = test_loader
        return test_loaders
    elif args.eval_dataset == 'cifar10_1':
        data, targets = load_cifar10_1('v4', args.datadir)
        preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize([0.5] * 3, [0.5] * 3)])
        test_transform = preprocess
        test_loaders = {}
        test_data = datasets.CIFAR10(args.datadir, train=False, transform=test_transform, download=True)
        test_data.data = data
        test_data.targets = torch.tensor(targets, dtype=torch.long)
        test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=args.batchsize,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True)
        return {default_subset: test_loader}
    elif args.eval_dataset == 'cifar5m':
        (X_te, Y_te), preproc = get_dataset('cifar5m', test_only=True)
        val_set = TransformingTensorDataset(X_te, Y_te, transform=preproc)
        test_loader = torch.utils.data.DataLoader(val_set, batch_size=256,
            shuffle=False, num_workers=args.workers, pin_memory=True)
        return {default_subset: test_loader}
    elif args.eval_dataset.startswith('cifar5m-binary'):
        (X_tr, Y_tr, X_te, Y_te), preproc = get_dataset(args.eval_dataset, test_only=True)
        val_set = TransformingTensorDataset(X_te, Y_te, transform=preproc)
        test_loader = torch.utils.data.DataLoader(val_set, batch_size=256,
            shuffle=False, num_workers=args.workers, pin_memory=True)
        return {default_subset: test_loader}

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

    if args.eval_dataset.startswith('cifar5m-binary'):
        nclasses = 2
    else:
        nclasses = 10

    # init logging
    logger = VanillaLogger(args, wandb, expanse_root=config['expanse_root'], hash=True)

    print('Loading dataset...')
    test_loaders = get_loaders()

    if args.eval_id_vs_ood:
        (X_te, Y_te), preproc = get_dataset('cifar5m', test_only=True)
        cf5m_val_set = TransformingTensorDataset(X_te, Y_te, transform=preproc)
        cf5m_test_loader = torch.utils.data.DataLoader(cf5m_val_set, batch_size=256,
            shuffle=False, num_workers=args.workers, pin_memory=True)
    print('Done loading.')

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda() if args.loss == 'xent' else mse_loss

    # order models based on step
    steps = []
    for f in glob.glob(args.pretrained + "**/model.pt", recursive=True):
        step = get_step(f)
        if step != -1:
            steps.append(get_step(f))
    steps.sort()

    for step in steps:
        if step < args.resume: #or step % 1024 != 0:
            continue

        f = f'{args.pretrained}/step{step:06}/model.pt'

        #load the model
        model = get_model32(args, args.arch, half=args.half, nclasses=nclasses, pretrained_path=f)
        if args.pretrained:
            load_state_dict(model, f)

        # model = torch.nn.DataParallel(model).cuda()
        if torch.cuda.is_available():
            model.cuda()

        print(f'Evaluating model {get_wandb_name(args)} at step {step}')

        d = {}
        if args.eval_id_vs_ood:
            test_cf5m = test_all(cf5m_test_loader, model, criterion, half=args.half, calibration_metrics=args.eval_calibration_metrics)
            d.update({f'CF-5m Test {k}' : v for k, v in test_cf5m.items()})

        if args.eval_dataset == 'cifar10c':
            mean_vals = defaultdict(list)

        d.update({'batch_num' : step})
        fname = f'{args.datadir}calibration/{wandb.run.id}_{step}.pickle'
        for name, test_loader in test_loaders.items():
            results = test_all(test_loader, model, criterion, half=args.half, calibration_metrics=args.eval_calibration_metrics, fname=fname)
            for k, v in results.items():
                print(f'{dataset_logs[args.eval_dataset]} {k} : {v}')

            if args.eval_dataset == 'cifar10c':
                for k, v in results.items():
                    d.update({ f'{dataset_logs[args.eval_dataset]} {name} {k}' : v})

                    mean_vals[f'{dataset_logs[args.eval_dataset]} {k}'].append(v)
            else:
                d.update({ f'{dataset_logs[args.eval_dataset]} {k}' : v for k, v in results.items()})


        if args.eval_dataset == 'cifar10c':
            mean_vals = {k: np.mean(v) for k, v in mean_vals.items()}
            d.update(mean_vals)

        if step != -1:
            logger.log_scalars(d, step=step)
        else:
            logger.wandb.summary.update(d)


if __name__ == '__main__':
    main()
