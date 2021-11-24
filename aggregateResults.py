import os
import pandas as pd
from pathlib import Path


def _aggregate(inpath, outpath, filename):
    dfs = []
    for file in os.listdir(inpath):
        if file.endswith(filename):
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

def calc_performanceGain(inpath, suffix):
    df_self = pd.read_csv(os.path.join(inpath, f"accuracy_selftrain_GC{suffix}.csv"), header=0, index_col=0)
    df_self = df_self[['test_acc_mean']]
    df_stats = pd.DataFrame()
    for algo in ['fedavg', 'fedprox_mu0.01', 'gcfl', 'gcflplus', 'gcflplusDWs']:
        tmp = pd.read_csv(os.path.join(inpath, f'accuracy_{algo}_GC{suffix}.csv'), index_col=0, header=0)
        tmp = tmp[['test_acc_mean']]
        df_diff = tmp - df_self
        df_stats.loc[algo, 'avg_acc_selftrain'] = df_self.mean().values[0]
        df_stats.loc[algo, 'avg_acc'] = tmp.mean().values[0]
        df_stats.loc[algo, 'avg_performanceGain'] = df_diff.mean().values[0]
        df_stats.loc[algo, 'std_performanceGain'] = df_diff.std().values[0]
        df_stats.loc[algo, 'max_performanceGain'] = df_diff.max().values[0]
        df_stats.loc[algo, 'min_performanceGain'] = df_diff.min().values[0]
        df_stats.loc[algo, 'perc_improved'] = df_diff[df_diff['test_acc_mean'] >= 0].count().values[0] * 1. / len(df_diff)
        df_stats.loc[algo, 'ratio'] = '{}/{}'.format(df_diff[df_diff['test_acc_mean'] >= 0].count().values[0], len(df_diff))
    print(df_stats)
    df_stats.to_csv(os.path.join(inpath, f"stats_performanceGain_GC{suffix}.csv"), header=True, index=True)

def average_aggregate_all(inpath, outpath, suffix):
    algos = ['selftrain', 'fedavg', 'fedprox_mu0.01', 'gcfl', 'gcflplus', 'gcflDWs']
    dfs = pd.DataFrame(index=algos, columns=['avg. of test_accuracy_mean', 'avg. of test_accuracy_std'])
    for algo in algos:
        df = pd.read_csv(os.path.join(inpath, f'accuracy_{algo}_GC{suffix}.csv'), header=0, index_col=0)
        if algo == 'selftrain':
            df = df[['test_acc_mean', 'test_acc_std']]
        dfs.loc[algo] = list(df.mean())
    # print(dfs)
    outfile = os.path.join(outpath, f'avg_accuracy_allAlgos_GC{suffix}.csv')
    dfs.to_csv(outfile, header=True, index=True)
    print("Wrote to:", outfile)

def main_aggregate_all_multiDS(inbase, outbase, datagroups, suffix):
    """ multiDS: aggregagte all outputs """
    for (data, hps) in datagroups:
        inpath = os.path.join(inbase, f'multiDS-nonOverlap/{data}/{hps}/repeats')
        outpath = os.path.join(outbase, f'multiDS-nonOverlap/{data}/{hps}')
        Path(outpath).mkdir(parents=True, exist_ok=True)
        for filename in ['accuracy_selftrain_GC.csv', 'accuracy_fedavg_GC.csv', 'accuracy_fedprox_mu0.01_GC.csv']:
            _aggregate(inpath, outpath, filename)
        for filename in ['accuracy_gcfl_GC.csv', 'accuracy_gcflplus_GC.csv', 'accuracy_gcflplusDWs_GC.csv']:
            _aggregate_cfl(inpath, outpath, filename)
        if suffix != '':
            for filename in [f'accuracy_selftrain_GC{suffix}.csv', f'accuracy_fedavg_GC{suffix}.csv', f'accuracy_fedprox_mu0.01_GC{suffix}.csv']:
                _aggregate(inpath, outpath, filename)
            for filename in [f'accuracy_gcfl_GC{suffix}.csv', f'accuracy_gcflplus_GC{suffix}.csv', f'accuracy_gcflplusDWs_GC{suffix}.csv']:
                _aggregate_cfl(inpath, outpath, filename)

    for (data, hps) in datagroups:
        print(data, hps)
        inpath = os.path.join(outbase, f'multiDS-nonOverlap/{data}/{hps}')
        calc_performanceGain(inpath, "")

    """ get average performance for all algorithms """
    for (data, hps) in datagroups:
        inpath = os.path.join(outbase, 'multiDS-nonOverlap', data, hps)
        average_aggregate_all(inpath, inpath, '')

def main_aggregate_all_oneDS(inbase, outbase, datagroups, suffix):
    for (data, hps) in datagroups:
        inpath = os.path.join(inbase, f'{data}/{hps}/repeats')
        outpath = os.path.join(outbase, f'{data}/{hps}')
        Path(outpath).mkdir(parents=True, exist_ok=True)
        for filename in ['accuracy_selftrain_GC.csv', 'accuracy_fedavg_GC.csv', 'accuracy_fedprox_mu0.01_GC.csv']:
            _aggregate(inpath, outpath, filename)
        for filename in ['accuracy_gcfl_GC.csv', 'accuracy_gcflplus_GC.csv', 'accuracy_gcflplusDWs_GC.csv']:
            _aggregate_cfl(inpath, outpath, filename)
        if suffix != '':
            for filename in [f'accuracy_selftrain_GC{suffix}.csv', f'accuracy_fedavg_GC{suffix}.csv', f'accuracy_fedprox_mu0.01_GC{suffix}.csv']:
                _aggregate(inpath, outpath, filename)
            for filename in [f'accuracy_gcfl_GC{suffix}.csv', f'accuracy_gcflplus_GC{suffix}.csv', f'accuracy_gcflplusDWs_GC{suffix}.csv']:
                _aggregate_cfl(inpath, outpath, filename)

    for (data, hps) in datagroups:
        print(data, hps)
        inpath = os.path.join(outbase, f'{data}/{hps}')
        calc_performanceGain(inpath, "")

    """ get average performance for all algorithms """
    for (data, hps) in datagroups:
        inpath = os.path.join(outbase, data, hps)
        average_aggregate_all(inpath, inpath, '')


if __name__ == '__main__':
    """ multiDS: aggregagte all outputs """
    inbase = f'/homelocal/hxie45/outputs/final3/seqLen{seq_len}/'
    outbase = f'/home/hxie45/priv/project/outputs/remote/final3/seqLen{seq_len}/'
    datagroups = [('molecules', 'eps_0.07_0.35'),
                  ('molecules', 'eps_0.07_0.28'), ('molecules', 'eps_0.08_0.32'), ('molecules', 'eps_0.09_0.36'), ('molecules', 'eps_0.1_0.4'),
                  ('biochem', 'eps_0.06_0.3'), ('biochem', 'eps_0.07_0.35'), ('biochem', 'eps_0.08_0.4'), ('biochem', 'eps_0.09_0.45'),
                  ('mix', 'eps_0.07_0.35'), ('mix', 'eps_0.08_0.4'), ('mix', 'eps_0.09_0.45'), ('mix', 'eps_0.1_0.5')]
    main_aggregate_all_multiDS(inbase, outbase, datagroups, '')
    # main_aggregate_all_multiDS(inbase, outbase, datagroups, '_degrs')

    """ oneDS: aggregagte all outputs """
    inbase = f'/local/scratch/hxie45/outputs/GCFL/seqLen{seq_len}/{overlap}/'
    outbase = f'/home/hxie45/priv/project/outputs/remote/final3/seqLen{seq_len}/{overlap}/'
    datagroups = [('NCI1-30clients', 'eps_0.04_0.08'), ('NCI1-30clients', 'eps_0.05_0.1'), ('NCI1-30clients', 'eps_0.06_0.1'), ('NCI1-30clients', 'eps_0.07_0.13'),
                  ('PROTEINS-10clients', 'eps_0.03_0.06'), ('PROTEINS-10clients', 'eps_0.04_0.07'), ('PROTEINS-10clients', 'eps_0.045_0.075'),
                  ('IMDB-BINARY-10clients', 'eps_0.025_0.045'), ('IMDB-BINARY-10clients', 'eps_0.03_0.05'), ('IMDB-BINARY-10clients', 'eps_0.035_0.06')]
    main_aggregate_all_oneDS(inbase, outbase, datagroups, '')
    # main_aggregate_all_oneDS(inbase, outbase, datagroups, '_degrs')
