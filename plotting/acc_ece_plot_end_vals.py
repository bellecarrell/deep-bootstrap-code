import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', type=str, help='wandb project name to plot', default='final')
    args = parser.parse_args()

    os.environ["WANDB_API_KEY"] = '5b28b0b43d10bf13a67459735a62e609dae35a19'

    api = wandb.Api()
    runs_str = f'deep-bootstrap2/real-ensemble'
    runs = api.runs(runs_str)
    dfs = []
    for run in runs:
        d = run.history()
        d["run"] = "real_ensemble" if run.id in ["3l2rfom7", "3m22kror"] else "real_single"
        d = d[d["batch_num"] == max(d["batch_num"])]
        if run.id != "2j30ieav":
            dfs.append(d)

    runs_str = f'deep-bootstrap2/ensemble'
    runs = api.runs(runs_str)
    for run in runs:
        d = run.history()
        d["run"] = "ideal_ensemble" if run.id in ["3l2rfom7", "3m22kror"] else "ideal_single"
        d = d[d["batch_num"] == max(d["batch_num"])]
        if run.id != "2j30ieav":
            dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    df['Test ece'] = np.where(df['Test ece'].isnull(), df['cifar5m-binary-hard ece'], df['Test ece'])
    df['Test SoftError'] = np.where(df['Test SoftError'].isnull(), df['cifar5m-binary-hard SoftError'], df['Test SoftError'])
    df['Test Error'] = np.where(df['Test Error'].isnull(), df['cifar5m-binary-hard Error'], df['Test Error'])
    df['Test SoftAcc'] = 1-df['Test SoftError']
    df['Test Acc'] = 1-df['Test Error']

    sns.scatterplot(data=df, x='Test SoftError', y='Test ece', hue='run')
    fname = f'{args.project}_soft.png'
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlim(0,0.4)
    plt.savefig(fname, bbox_inches='tight')
    plt.clf()        
    sns.scatterplot(data=df, x='Test Error', y='Test ece', hue='run')
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlim(0,0.4)
    fname = f'{args.project}.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.clf()  

    sns.scatterplot(data=df, x='Test SoftAcc', y='Test ece', hue='run')
    fname = f'{args.project}_softacc.png'
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.savefig(fname, bbox_inches='tight')
    plt.clf()        
    sns.scatterplot(data=df, x='Test Acc', y='Test ece', hue='run')
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fname = f'{args.project}_acc.png'
    plt.savefig(fname, bbox_inches='tight')

if __name__=='__main__':
    main()
