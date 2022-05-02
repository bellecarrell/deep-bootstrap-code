import argparse
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import seaborn as sns
import wandb

def interval_right(row, col):
    if type(row[col]) == pd._libs.interval.Interval:
        return row[col].mid
    else:
        return row[col]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', type=str, help='wandb project name to plot', default='real-ensemble')
    args = parser.parse_args()

    os.environ["WANDB_API_KEY"] = '5b28b0b43d10bf13a67459735a62e609dae35a19'

    api = wandb.Api()
    runs_str = f'deep-bootstrap2/{args.project}'
    runs = api.runs(runs_str)
    dfs = []
    for run in runs:
        d = run.history()
        d["run"] = "ensemble" if run.id in ["3l2rfom7", "3m22kror"] else "single"
        if run.id != "2j30ieav":
            dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    df['Test ece'] = np.where(df['Test ece'].isnull(), df['cifar5m-binary-hard ece'], df['Test ece'])
    df['Test Loss'] = np.where(df['Test Loss'].isnull(), df['cifar5m-binary-hard Loss'], df['Test Loss'])
    df['Test SoftError'] = np.where(df['Test SoftError'].isnull(), df['cifar5m-binary-hard SoftError'], df['Test SoftError'])
    df['Test Error'] = np.where(df['Test Error'].isnull(), df['cifar5m-binary-hard Error'], df['Test Error'])
    df['Test SoftAcc'] = 1-df['Test SoftError']
    df['Test Acc'] = 1-df['Test Error']
    bins = np.arange(0.45,1.0,0.02)
    df['Test SoftAccBinned'] = pd.cut(df['Test SoftAcc'], bins)
    df['Test SoftAccBinned'] = df.apply(lambda row: interval_right(row, 'Test SoftAccBinned'), axis=1)
    bins = np.arange(0.45,1.0,0.02)
    df['Test AccBinned'] = pd.cut(df['Test Acc'], bins)
    df['Test AccBinned'] = df.apply(lambda row: interval_right(row, 'Test AccBinned'), axis=1)

    bins = np.arange(0.02,0.4,0.02)
    df['Test SoftErrorBinned'] = pd.cut(df['Test SoftError'], bins)
    df['Test SoftErrorBinned'] = df.apply(lambda row: interval_right(row, 'Test SoftErrorBinned'), axis=1)
    df['Test ErrorBinned'] = pd.cut(df['Test Error'], bins)
    df['Test ErrorBinned'] = df.apply(lambda row: interval_right(row, 'Test ErrorBinned'), axis=1)

    bins = np.arange(0,1.7,0.05)
    df['Test Loss'] = pd.cut(df['Test Loss'], bins)
    df['Test Loss'] = df.apply(lambda row: interval_right(row, 'Test Loss'), axis=1)


    sns.lineplot(data=df, x='Test SoftErrorBinned', y='Test ece', hue='run')
    fname = f'{args.project}_soft.png'
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlim(0,0.4)
    plt.title(fname)
    plt.savefig(fname, bbox_inches='tight')
    plt.clf()        
    sns.lineplot(data=df, x='Test ErrorBinned', y='Test ece', hue='run')
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlim(0,0.4)
    fname = f'{args.project}.png'
    plt.title(fname)
    plt.savefig(fname, bbox_inches='tight')
    plt.clf()  

    sns.lineplot(data=df, x='Test SoftAccBinned', y='Test ece', hue='run')
    fname = f'{args.project}_softacc.png'
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.title(fname)
    plt.savefig(fname, bbox_inches='tight')
    plt.clf()        
    sns.lineplot(data=df, x='Test AccBinned', y='Test ece', hue='run')
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fname = f'{args.project}_acc.png'
    plt.title(fname)
    plt.savefig(fname, bbox_inches='tight')
    plt.clf() 
    sns.lineplot(data=df, x='Test Loss', y='Test ece', hue='run')
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fname = f'{args.project}_loss.png'
    plt.title(fname)
    plt.savefig(fname, bbox_inches='tight')

if __name__=='__main__':
    main()
