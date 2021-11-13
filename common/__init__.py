import torch
import torchvision.datasets as ds
import numpy as np
from torchvision.transforms import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm
from .datasets import dload, download_dir
import subprocess

import pickle
import glob as pyglob

def gsave(x, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(x, f)

def gload(filepath):
    with open(filepath, 'rb') as f:
        x = pickle.load(f)
    return x

def glob(filepath):
    return pyglob.glob(filepath)

def save_model(model, filepath):
    def unwrap_model(model): # unwraps DataParallel, etc
        return model.module if hasattr(model, 'module') else model
    '''
    If filepath is passed saving a .pt with custom naming already
    then use that, otherwise automatically add model.pt
    '''
    if not filepath.endswith('.pt'):
        local_path = f'{filepath}/model.pt'
    else:
        local_path = filepath
    torch.save(unwrap_model(model).state_dict(), local_path)

def load_state_dict(model, filepath, crc=False, filter_mismatched_keys=False):
    #local_path = dload(filepath, overwrite=True, crc=crc)
    local_path = filepath # TODO: any other validations?
    if torch.cuda.is_available():
        pretrained_dict = torch.load(local_path)
    else:
        pretrained_dict = torch.load(local_path, map_location=torch.device('cpu'))
    if filter_mismatched_keys:
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def predict(model, X, bs=256, dev='cuda:0'):
    yhat = torch.empty(len(X), dtype=torch.long).to(dev)

    model.eval()
    model.to(dev)
    with torch.no_grad():
        for i in range((len(X)-1)//bs + 1):
            xb = X[i*bs : i*bs+bs].to(dev)
            outputs = model(xb)
            _, preds = torch.max(outputs, dim=1)
            yhat[i*bs : i*bs+bs] = preds

    return yhat.cpu()

def predict_ds(model, ds: Dataset, bsize=128):
    ''' Returns loss, acc'''
    test_dl = DataLoader(ds, batch_size=bsize, shuffle=False, num_workers=4)

    model.eval()
    model.cuda()
    allPreds = []
    with torch.no_grad():
        for (xb, yb) in tqdm(test_dl):
            xb, yb = xb.cuda(), yb.cuda()
            outputs = model(xb)
            preds = torch.argmax(outputs[1], dim=1)
            allPreds.append(preds)

    preds = torch.cat(allPreds).long().cpu().numpy().astype(np.uint8)
    return preds

def evaluate(model, X, Y, bsize=512, loss_func=nn.CrossEntropyLoss().cuda()):
    ''' Returns loss, acc'''
    ds = TensorDataset(X, Y)
    test_dl = DataLoader(ds, batch_size=bsize, shuffle=False, num_workers=1)

    model.eval()
    model.cuda()
    nCorrect = 0.0
    nTotal = 0
    net_loss = 0.0
    with torch.no_grad():
        for (xb, yb) in test_dl:
            xb, yb = xb.cuda(), yb.cuda()
            outputs = model(xb)
            loss = len(xb) * loss_func(outputs, yb)
            _, preds = torch.max(outputs, dim=1)
            nCorrect += (preds == yb).float().sum()
            net_loss += loss
            nTotal += preds.size(0)

    acc = nCorrect.cpu().item() / float(nTotal)
    loss = net_loss.cpu().item() / float(nTotal)
    return loss, acc
