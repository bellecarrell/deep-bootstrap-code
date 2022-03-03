import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import argparse
import os
import wandb
import numpy as np

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

    exps = [(5000, 'False'), (10000, 'False'), (25000, 'False'), (50000, 'False'), (50000, 'True')]
    colors = cm.jet(np.linspace(0,1,len(exps)))
    data = {}
    for run in runs_df['summary']:
        if (run['pretrain_n'], run['pretrain_iid']) not in data:
            data[(run['pretrain_n'], run['pretrain_iid'])] = {}
        data[(run['pretrain_n'], run['pretrain_iid'])][run['pretrain_step']] = (run['Test SoftError'], run['Test Error'])
    for color_idx, exp in enumerate(exps):
        keys = list(data[exp].keys())
        sort_idx = np.argsort(keys)
        x_ax = [keys[idx] for idx in sort_idx]
        y_ax_softerror = [data[exp][key_idx][0] for key_idx in x_ax]
        y_ax_error = [data[exp][key_idx][1] for key_idx in x_ax]
        plt.plot(x_ax, y_ax_softerror, color=colors[color_idx], label=f'pretrain n={exp[0]}, ideal={exp[1]=="True"}')

    plt.legend()
    plt.xlabel('pretrain step')
    plt.ylabel('CF100 test finetune soft error')
    plt.title('CF100 e2e fine-tuning (lr=1e-3, 10 epochs, cosine annealing)')
    plt.show()

if __name__=='__main__':
    main();
