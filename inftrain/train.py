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

from .utils import AverageMeter
from common.datasets import load_cifar, TransformingTensorDataset, get_cifar_data_aug, AugMixDataset
from common.datasets import load_cifar550, load_svhn_all, load_svhn, load_cifar5m, load_cifar100, load_pacs
import common.models32 as models
from .utils import get_model32, get_optimizer, get_scheduler, make_loader_cifar10_1, get_wandb_name, get_dataset, add_noise, get_data_aug, cuda_transfer, recycle, mse_loss, test_all, augmentations

from common.logging import VanillaLogger

parser = argparse.ArgumentParser(description='vanilla training')
parser.add_argument('--proj', default='test-soft', type=str, help='project name')
parser.add_argument('--dataset', default='cifar5m', type=str)
parser.add_argument('--datadir', type=str, default='', help='path to data')
parser.add_argument('--nsamps', default=50000, type=int, help='num. train samples')
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--k', default=64, type=int, help="log every k batches", dest='k')
parser.add_argument('--save-at-k', default=False, action='store_true', help='save model at every k step (and more often in early stages)')
parser.add_argument('--iid', default=False, action='store_true', help='simulate infinite samples (fresh samples each batch)')
parser.add_argument('--save_model_step', default=-1, type=int, help='step frequency for saving intermediate models')

# parser.add_argument('--arch', metavar='ARCH', default='mlp[16384,16384,512]')
parser.add_argument('--arch', metavar='ARCH', default='preresnet18')
parser.add_argument('--pretrained', type=str, default=None, help='expanse path to pretrained model state dict (optional)')
parser.add_argument('--width', default=None, type=int, help="architecture width parameter (optional)")
parser.add_argument('--loss', default='xent', choices=['xent', 'mse'], type=str)
parser.add_argument('--cifar10-1', default=False, action='store_true', help='evaluate on cifar10.1')

parser.add_argument('--opt', default="sgd", type=str)
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--scheduler', default="cosine", type=str, help='lr scheduler')
parser.add_argument('--sched', default=None, type=str)
parser.add_argument('--aug', default=0, type=int, help='data-aug (0: none, 1: flips, 2: all)')
parser.add_argument('--augmix', default=False, action='store_true', help='perform AugMix')
# AugMix options
parser.add_argument(
    '--mixture-width',
    default=3,
    type=int,
    help='Number of augmentation chains to mix per augmented example')
parser.add_argument(
    '--mixture-depth',
    default=-1,
    type=int,
    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument(
    '--aug-severity',
    default=3,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--no-jsd',
    '-nj',
    action='store_true',
    help='Turn off JSD consistency loss.')
parser.add_argument(
    '--all-ops',
    '-all',
    action='store_true',
    help='Turn on all operations (+brightness,contrast,color,sharpness).')
parser.add_argument('--epochs', default=100, type=int)
# for keeping the same LR sched across different samp sizes.
parser.add_argument('--nbatches', default=None, type=int, help='Total num. batches to train for. If specified, overrides EPOCHS.')
parser.add_argument('--batches_per_lr_step', default=390, type=int)

parser.add_argument('--noise', default=0.0, type=float, help='label noise probability (train & test).')

parser.add_argument('--momentum', default=0.0, type=float, help='momentum (0 or 0.9)')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay')

parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--half', default=False, action='store_true', help='training with half precision')
parser.add_argument('--fast', default=False, action='store_true', help='do not log more frequently in early stages')
parser.add_argument('--earlystop', default=False, action='store_true', help='stop when train loss < 0.01')

parser.add_argument('--aseed', default=None, type=int, help="architecture-seed for rand NAS archs (optional)")
parser.add_argument('--comment', default=None)

args = parser.parse_args()

def make_loader(x, y, transform=None, batch_size=256):
    dataset = TransformingTensorDataset(x, y, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            shuffle=False, num_workers=args.workers, pin_memory=True)
    return loader

def aug(image, preprocess):
  """Perform AugMix augmentations and compute mixture.
  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.
  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations
  if args.all_ops:
    aug_list = augmentations.augmentations_all

  ws = np.float32(np.random.dirichlet([1] * args.mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(args.mixture_width):
    image_aug = image.copy()
    depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, args.aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed


def main():
    ## argparsing hacks
    if args.sched is not None:
        sched = list(map(int, args.sched.split(',')))
        args.epochs = sum(sched)
        args.scheduler = 'steps'
    if args.pretrained == 'None':
        args.pretrained = None # hack for caliban

    wandb.init(project=args.proj, entity='deep-bootstrap2')
    wandb.run.name = wandb.run.id  + " - " + get_wandb_name(args)
    wandb.run.save()
    cudnn.benchmark = True

    #load the model
    model = get_model32(args, args.arch, half=args.half, nclasses=10, pretrained_path=args.pretrained)
    # model = torch.nn.DataParallel(model).cuda()
    if torch.cuda.is_available():
        model.cuda()

    # init logging
    logger = VanillaLogger(args, wandb, hash=True)

    print('Loading datasets...')

    (X_tr, Y_tr, X_te, Y_te), preproc = get_dataset(args.dataset)

    # subsample
    if not args.iid:
        I = np.random.permutation(len(X_tr))[:args.nsamps]
        X_tr, Y_tr = X_tr[I], Y_tr[I]

    # Add noise (optionally)
    Y_tr = add_noise(Y_tr, args.noise)
    Y_te = add_noise(Y_te, args.noise)

    if args.augmix:
        train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4), 
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)])
        
        preprocess = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)])
        test_transform = preprocess

        tr_set = AugMixDataset(X_tr, Y_tr, preproc, args)

    else:
        tr_set = TransformingTensorDataset(X_tr, Y_tr, transform=transforms.Compose([preproc, get_data_aug(args.aug)]))
    
    val_set = TransformingTensorDataset(X_te, Y_te, transform=preproc)

    tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=args.batchsize,
            shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True) # drop the last batch if it's incomplete (< batch size)
    te_loader = torch.utils.data.DataLoader(val_set, batch_size=256,
            shuffle=False, num_workers=args.workers, pin_memory=True)

    cifar_test = make_loader(*(load_cifar()[2:])) # original cifar-10 test set

    if args.cifar10_1:
        cifar10_1_loader = make_loader_cifar10_1(args)

    print('Done loading.')


    # batches / lr computations
    batches_per_epoch = int(np.floor(args.nsamps / args.batchsize))
    if args.nbatches is None:
        # set nbatches from EPOCHS
        args.nbatches = int((args.nsamps / args.batchsize) * args.epochs)
        args.batches_per_lr_step = batches_per_epoch
    num_lr_steps = (args.nbatches) // args.batches_per_lr_step # = epochs (unless --epochs is overridden)
    print(f'Num. total train batches: {args.nbatches}')


    # define loss function (criterion), optimizer and scheduler
    criterion = nn.CrossEntropyLoss().cuda() if args.loss == 'xent' else mse_loss
    optimizer = get_optimizer(args.opt, model.parameters(), args.lr, args.momentum, args.wd)
    scheduler = get_scheduler(args, args.scheduler, optimizer, num_epochs=num_lr_steps, batches_per_epoch=args.batches_per_lr_step)



    n_tot = 0
    for i, (images, target) in enumerate(recycle(tr_loader)):
        model.train()
        if torch.cuda.is_available():
            images, target = cuda_transfer(images, target)

        if args.augmix:
            if args.no_jsd:
                #images = images.cuda()
                #targets = targets.cuda()
                logits = model(images)
                loss = F.cross_entropy(logits, target)
            else:
                images_all = torch.cat(images, 0).cuda()
                #targets = targets.cuda()
                logits_all = model(images_all)
                logits_clean, logits_aug1, logits_aug2 = torch.split(
                    logits_all, images[0].size(0))

                # Cross-entropy is only computed on clean images
                loss = F.cross_entropy(logits_clean, target)

                p_clean, p_aug1, p_aug2 = F.softmax(
                    logits_clean, dim=1), F.softmax(
                        logits_aug1, dim=1), F.softmax(
                            logits_aug2, dim=1)

                # Clamp mixture distribution to avoid exploding KL divergence
                p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
                loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                                F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                                F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

        else:
            output = model(images)
            loss = criterion(output, target)

        n_tot += len(images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## logging
        lr = optimizer.param_groups[0]['lr']

        # if i % args.k == 0: # first 512 batches, and every kth batch after that
        if i % args.k == 0 or (not args.fast and  ( \
            # (i < 128) or \
            # (i < 512 and i % 2 == 0) or \
            (i < 1024 and i % 4 == 0) or \
            (i < 2048 and i % 8 == 0))):
            ''' Every k batches (and more frequently in early stages): log train/test errors. '''

            d = {'batch_num': i,
                'lr': lr,
                'n' : n_tot}

            test_m = test_all(te_loader, model, criterion)
            testcf_m = test_all(cifar_test, model, criterion)
            d.update({ f'Test {k}' : v for k, v in test_m.items()})
            d.update({ f'CF10 {k}' : v for k, v in testcf_m.items()})

            if args.cifar10_1:
                cf10_1_m = test_all(cifar10_1_loader, model, criterion)
                d.update({ f'CF10.1 {k}' : v for k, v in cf10_1_m.items()})

            if not args.iid:
                train_m = test_all(tr_loader, model, criterion)
                d.update({ f'Train {k}' : v for k, v in train_m.items()})

                print(f'Batch {i}.\t lr: {lr:.3f}\t Train Loss: {d["Train Loss"]:.4f}\t Train Error: {d["Train Error"]:.3f}\t Test Error: {d["Test Error"]:.3f}')
            else:
                print(f'Batch {i}.\t lr: {lr:.3f}\t Test Error: {d["Test Error"]:.3f}')


            logger.log_scalars(d)
            logger.flush()

            if args.save_at_k:
                logger.save_model_step(i, model)

        if args.save_model_step > 0 and (i+1) % args.save_model_step == 0:
            logger.save_model_step(i, model)

        if (i+1) % args.batches_per_lr_step == 0:
            scheduler.step()

        if (i+1) % batches_per_epoch == 0:
            print(f'[ Epoch {i // batches_per_epoch} ]')

        if (i+1) == args.nbatches:
            break;

        if not args.iid and args.earlystop and d['Train Loss'] < 0.01:
            break; # break if small train loss

    ## Final logging
    logger.save_model(model)

    summary = {}
    summary.update({ f'Final Test {k}' : v for k, v in test_all(te_loader, model, criterion).items()})
    summary.update({ f'Final Train {k}' : v for k, v in test_all(tr_loader, model, criterion).items()})
    summary.update({ f'Final CF10 {k}' : v for k, v in test_all(cifar_test, model, criterion).items()})

    if args.cifar10_1:
        summary.update({ f'Final CF10.1 {k}' : v for k, v in test_all(cifar10_1_loader, model, criterion).items()})

    logger.log_summary(summary)
    logger.flush()


if __name__ == '__main__':
    main()
