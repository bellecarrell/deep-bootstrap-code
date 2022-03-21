import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb

# parser = argparse.ArgumentParser(description='vanilla training')
# parser.add_argument('--file', default='/home/annabelle/workplace/summary.pickle', type=str, help='summary file to make plot from')

# args = parser.parse_args()

d = {'soft_error': [], 'ece': [], 'sce': [], 'key': [], 'run': []}


def main():

    api = wandb.Api()

    runs = api.runs("deep-bootstrap2/ideal-idece")

    for run in runs:
        run_d = {'soft_error': [], 'ece': [], 'sce': [], 'key': []}
        print('here')

        for i, vals in run.history().iterrows():
            print(vals)
            for k in run.history().columns:
                print(k)
                if k.endswith("ece"):
                    soft_error_key = k.strip("ece") + "SoftError"
                    sce_key = k.strip("ece") + "sce"
                    run_d['ece'].append(vals[k])
                    run_d['soft_error'].append(vals[soft_error_key])
                    run_d['sce'].append(vals[sce_key])
                    run_d['key'].append(k.strip(" ece"))

                    if i == len(run.history()) - 1:
                        d['ece'].append(vals[k])
                        d['soft_error'].append(vals[soft_error_key])
                        d['sce'].append(vals[sce_key])
                        key = k.strip(" ece")
                        d['key'].append(key)
                        d['run'].append(run.id)

        df = pd.DataFrame(data=run_d)
        print(df.head())
        sns.scatterplot(data=df, x='soft_error', y='ece', hue='key')
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
        fname = run.id + '_ece.png'
        plt.savefig(fname, bbox_inches='tight')
        plt.clf()        
        sns.scatterplot(data=df, x='soft_error', y='sce', hue='key')
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
        fname = run.id + '_sce.png'
        plt.savefig(fname, bbox_inches='tight')
        plt.clf()
            

    df = pd.DataFrame(data=d)

    sns.scatterplot(data=df, x='soft_error', y='ece', hue='key', style='run')
    fname = 'all_ece.png'
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.savefig(fname, bbox_inches='tight')
    plt.clf()        
    sns.scatterplot(data=df, x='soft_error', y='sce', hue='key', style='run')
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fname = 'all_sce.png'
    plt.savefig(fname, bbox_inches='tight')

if __name__ == '__main__':
    main()
