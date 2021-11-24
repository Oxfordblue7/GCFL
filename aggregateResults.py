import os
import argparse
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
    print(f"Performance Gain:\ndf_stats")
    df_stats.to_csv(os.path.join(inpath, f"stats_performanceGain_GC{suffix}.csv"), header=True, index=True)

def average_aggregate_all(inpath, outpath, suffix):
    algos = ['selftrain', 'fedavg', 'fedprox_mu0.01', 'gcfl', 'gcflplus', 'gcflDWs']
    dfs = pd.DataFrame(index=algos, columns=['avg. of test_accuracy_mean', 'avg. of test_accuracy_std'])
    for algo in algos:
        df = pd.read_csv(os.path.join(inpath, f'accuracy_{algo}_GC{suffix}.csv'), header=0, index_col=0)
        if algo == 'selftrain':
            df = df[['test_acc_mean', 'test_acc_std']]
        dfs.loc[algo] = list(df.mean())
    outfile = os.path.join(outpath, f'avg_accuracy_allAlgos_GC{suffix}.csv')
    dfs.to_csv(outfile, header=True, index=True)
    print("Wrote to:", outfile)

def main_aggregate_all_multiDS(inpath, outpath, suffix):
    """ multiDS: aggregagte all outputs """
    Path(outpath).mkdir(parents=True, exist_ok=True)
    for filename in ['accuracy_selftrain_GC.csv', 'accuracy_fedavg_GC.csv', 'accuracy_fedprox_mu0.01_GC.csv',
                     'accuracy_gcfl_GC.csv', 'accuracy_gcflplus_GC.csv', 'accuracy_gcflplusDWs_GC.csv']:
        _aggregate(inpath, outpath, filename)
    if suffix != '':
        for filename in [f'accuracy_selftrain_GC{suffix}.csv', f'accuracy_fedavg_GC{suffix}.csv', f'accuracy_fedprox_mu0.01_GC{suffix}.csv',
                         f'accuracy_gcfl_GC{suffix}.csv', f'accuracy_gcflplus_GC{suffix}.csv', f'accuracy_gcflplusDWs_GC{suffix}.csv']:
            _aggregate(inpath, outpath, filename)

    calc_performanceGain(inpath, "")

    """ get average performance for all algorithms """
    average_aggregate_all(inpath, inpath, '')

def main_aggregate_all_oneDS(inpath, outpath, suffix):
    Path(outpath).mkdir(parents=True, exist_ok=True)
    for filename in ['accuracy_selftrain_GC.csv', 'accuracy_fedavg_GC.csv', 'accuracy_fedprox_mu0.01_GC.csv',
                     'accuracy_gcfl_GC.csv', 'accuracy_gcflplus_GC.csv', 'accuracy_gcflplusDWs_GC.csv']:
        _aggregate(inpath, outpath, filename)
    if suffix != '':
        for filename in [f'accuracy_selftrain_GC{suffix}.csv', f'accuracy_fedavg_GC{suffix}.csv', f'accuracy_fedprox_mu0.01_GC{suffix}.csv',
                         f'accuracy_gcfl_GC{suffix}.csv', f'accuracy_gcflplus_GC{suffix}.csv', f'accuracy_gcflplusDWs_GC{suffix}.csv']:
            _aggregate(inpath, outpath, filename)

    calc_performanceGain(inpath, "")

    """ get average performance for all algorithms """
    average_aggregate_all(inpath, inpath, '')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='./outputs',
                        help='The input path of the experimental results.')
    parser.add_argument('--outpath', type=str, default='./outputs',
                        help='The out path for outputting.')

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    """ multiDS: aggregagte all outputs """
    main_aggregate_all_multiDS(args.inpath, args.outpath, '')
    # main_aggregate_all_multiDS(args.inpath, args.outpath, '_degrs')

    """ oneDS: aggregagte all outputs """
    main_aggregate_all_oneDS(args.inpath, args.outpath, '')
    # main_aggregate_all_oneDS(args.inpath, args.outpath, '_degrs')
