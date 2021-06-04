import os
import re
import pandas as pd
from pathlib import Path

def _aggregate(inpath, outpath, filename):
    dfs = []
    for file in os.listdir(inpath):
        if file.endswith(filename):
        # if re.match('\d_{}'.format(filename), file):
            dfs.append(pd.read_csv(os.path.join(inpath, file), header=0, index_col=0))
    df = pd.concat(dfs)
    group = df.groupby(df.index)
    dfmean = group.mean()
    dfstd = group.std()
    df_out = dfmean.join(dfstd, lsuffix='_mean', rsuffix='_std')
    df_out.to_csv(os.path.join(outpath, filename), header=True, index=True)


def _aggregate_cfl(inpath, outpath, filename):
    dfs = []
    for file in os.listdir(inpath):
        if file.endswith(filename):
        # if re.match('\d_{}'.format(filename), file):
            tmp = pd.read_csv(os.path.join(inpath, file), header=0, index_col=0)
            tmp = pd.DataFrame(tmp.max(axis=1))
            tmp.columns = ['test_acc']
            dfs.append(tmp)

    df = pd.concat(dfs)
    group = df.groupby(df.index)
    dfmean = group.mean()
    dfstd = group.std()
    df_out = dfmean.join(dfstd, lsuffix='_mean', rsuffix='_std')
    df_out.to_csv(os.path.join(outpath, filename), header=True, index=True)


def average_aggregate_all(inpath, outpath, suffix):
    algos = ['selftrain', 'fedavg', 'fedprox_mu0.01', 'gcfl', 'gcfldtw', 'gcfldtwDWs']
    dfs = pd.DataFrame(index=algos, columns=['avg. of test_accuracy_mean', 'avg. of test_accuracy_std'])
    for algo in algos:
        df = pd.read_csv(os.path.join(inpath, f'accuracy_{algo}{suffix}.csv'), header=0, index_col=0)
        dfs.loc[algo] = list(df.mean())
    # print(dfs)
    outfile = os.path.join(outpath, f'avg_accuracy_allAlgos{suffix}.csv')
    dfs.to_csv(outfile, header=True, index=True)
    print("Wrote to:", outfile)


def calc_performanceGain(inpath, suffix):
    df_self = pd.read_csv(os.path.join(inpath, f"accuracy_selftrain{suffix}.csv"), header=0, index_col=0)
    df_self = df_self[['test_acc_mean']]
    df_stats = pd.DataFrame()
    for algo in ['fedavg', 'fedprox_mu0.01', 'gcfl', 'gcfldtw', 'gcfldtwDWs']:
        tmp = pd.read_csv(os.path.join(inpath, f'accuracy_{algo}{suffix}.csv'), index_col=0, header=0)
        tmp = tmp[['test_acc_mean']]
        df_diff = tmp - df_self
        df_stats.loc[algo, 'avg_performanceGain'] = df_diff.mean().values[0]
        df_stats.loc[algo, 'std_performanceGain'] = df_diff.std().values[0]
        df_stats.loc[algo, 'max_performanceGain'] = df_diff.max().values[0]
        df_stats.loc[algo, 'min_performanceGain'] = df_diff.min().values[0]
        df_stats.loc[algo, 'perc_improved'] = df_diff[df_diff['test_acc_mean'] >= 0].count().values[0] * 1. / len(df_diff)
    # print(df_stats)
    outfile = os.path.join(inpath, f"stats_performanceGain{suffix}.csv")
    df_stats.to_csv(outfile, header=True, index=True)
    print("Wrote to:", outfile)


def main_aggregate(inbase='./outputs', outbase='./outputs'):
    # oneDS
    for (data, numClients) in [('PROTEINS', 30), ('NCI1', 80), ('IMDB-BINARY', 20)]:
        inpath = os.path.join(inbase, f'oneDS-nonOverlap-{numClients}clients/{data}/repeats')
        outpath = os.path.join(outbase, f'oneDS-nonOverlap-{numClients}clients/{data}')
        Path(outpath).mkdir(parents=True, exist_ok=True)
        for filename in ['accuracy_selftrain.csv', 'accuracy_fedavg.csv', 'accuracy_fedprox_mu0.01.csv']:
            _aggregate(inpath, outpath, filename)
        for filename in ['accuracy_gcfl.csv', 'accuracy_gcfldtw.csv', 'accuracy_gcfldtwDWs.csv']:
            _aggregate_cfl(inpath, outpath, filename)

        average_aggregate_all(outpath, outpath, '')

        calc_performanceGain(outpath, '')

    # multiDS
    for data in ['molecules', 'biochem', 'mix']:
        inpath = os.path.join(inbase, f'multiDS-nonOverlap/{data}/repeats')
        outpath = os.path.join(outbase, f'multiDS-nonOverlap/{data}')
        Path(outpath).mkdir(parents=True, exist_ok=True)
        for filename in ['accuracy_selftrain.csv', 'accuracy_fedavg.csv', 'accuracy_fedprox_mu0.01.csv']:
            _aggregate(inpath, outpath, filename)
        for filename in ['accuracy_gcfl.csv', 'accuracy_gcfldtw.csv', 'accuracy_gcfldtwDWs.csv']:
            _aggregate_cfl(inpath, outpath, filename)

        average_aggregate_all(outpath, outpath, '')

        calc_performanceGain(outpath, '')


if __name__ == '__main__':
    main_aggregate()