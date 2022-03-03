import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from adjustText import adjust_text
import pandas as pd
import argparse
import os
import wandb
import numpy as np

aug_label_map = {
    0: 'none',
    1: 'flips',
    2: 'flips+crops',
    3: 'flips+reflect-crop',
    4: 'crops',
    5: 'augmix'
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity', '-e', type=str, default='belkinlab', help='wandb entity')
    parser.add_argument('--project', '-p', type=str, required=True, help='wandb project name to plot')
    args = parser.parse_args()

    os.environ["WANDB_API_KEY"] = '5b28b0b43d10bf13a67459735a62e609dae35a19'
    api = wandb.Api()

    runs = api.runs(args.entity + "/" + args.project)

    summary_list, config_list = [], []
    for run in runs:
        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k,v in run.config.items()
         if not k.startswith('_')})

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list
    })

    data = {}
    for summary, config in zip(runs_df['summary'], runs_df['config']):
        if 'CF-5m Test SoftError' in summary:
            data[(config['aug'], config['iid'], config['nsamps'])] = (
                summary['CF10.1 SoftError'],
                summary['CF10.1 Error'],
                summary['CF-5m Test SoftError'],
                summary['CF-5m Test Error']
                )

    augs = [0, 2, 5]
    nsamps = [5000, 10000, 25000, 50000, 50000]
    iid = [False, False, False, False, True]
    colors = cm.jet(np.linspace(0,1,len(nsamps)))

    texts = []
    xax = []
    yax = []
    xmin = 1.0
    xmax = 0.0
    for color_idx, (nsamp, ideal) in enumerate(zip(nsamps, iid)):
        for aug in augs:
            res = data[(aug, ideal, nsamp)]
            xax.append(1-res[2])
            yax.append(1-res[0])
            xmin = min(xmin, 1-res[2])
            xmax = max(xmax, 1-res[2])
            plt.scatter(1-res[2], 1-res[0], color=colors[color_idx])
            texts.append(plt.text(1-res[2], 1-res[0], f'{aug_label_map[aug]}', fontsize='medium'))
            #plt.annotate(f'{"ideal" if ideal==True else "real"}, {aug_label_map[aug]}', (res[0], res[2]), \
            #             xytext=(-80, 10), textcoords='offset points', arrowprops=dict(arrowstyle = '-', connectionstyle = 'arc3'))
    adjust_text(texts)
    z = np.polyfit(xax, yax, 1)
    # z2 = np.polyfit(xax, yax, 2)
    xax = np.linspace(xmin, xmax, 100)
    yax = xax*z[0] + z[1]
    # yax2 = np.power(xax, 2)*z2[0] + xax*z2[1] + z2[2]
    plt.plot(xax, yax, color='black')
    # plt.plot(xax, yax2, color='blue')
    # fit_legends = [f'{z[0]:.2f}x + {z[1]:.2f}', f'{z2[0]:.2f}x^2 + {z2[1]:.2f}x + {z2[0]:.2f}']
    fit_legends = [f'{z[0]:.2f}x + {z[1]:.2f}']
    # fit_legends = []
    plt.legend(fit_legends + [f'{"ideal" if ideal==True else "n="+str(n)}' for n,ideal in zip(nsamps, iid)])
    ax = plt.gca()
    leg = ax.get_legend()
    for color_idx in range(len(nsamps)):
        leg.legendHandles[color_idx+len(fit_legends)].set_color(colors[color_idx])
    plt.ylabel('CIFAR 10.1 Test Soft Acc')
    plt.xlabel('CIFAR 5m Test Soft Acc')
    plt.title('OOD vs. ID Test Soft Acc')
    plt.show()

if __name__=='__main__':
    main();
