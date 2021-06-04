import os
import argparse
import random

import torch
from pathlib import Path

import setupGC
import fl_setupGC
from training import *


def process_selftrain(local_epoch):
    clients, server, idx_clients = setupGC.setup_devices(splitedData, args)
    print("\nDone setting up self-training devices.")

    print("Self-training ...")
    df = pd.DataFrame()
    accs = run_selftrain_GC(clients, server, local_epoch)
    for k, v in accs.items():
        df.loc[k, 'test_acc'] = v
    # print(df)
    if args.repeat is None:
        outfile = os.path.join(outpath, args.data_group, f'accuracy_selftrain{suffix}.csv')
    else:
        outfile = os.path.join(outpath, args.data_group, "repeats", f'{args.repeat}_accuracy_selftrain{suffix}.csv')
    df.to_csv(outfile)
    print(f"Wrote to: {outfile}")

def process_fedavg():
    clients, server, idx_clients = setupGC.setup_devices(splitedData, args)
    print("\nDone setting up FedAvg devices.")

    print("Running FedAvg ...")
    frame = run_fedavg(clients, server, args.num_rounds, args.local_epoch, samp=None)
    if args.repeat is None:
        outfile = os.path.join(outpath, args.data_group, f'accuracy_fedavg{suffix}.csv')
    else:
        outfile = os.path.join(outpath, args.data_group, "repeats", f'{args.repeat}_accuracy_fedavg{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to: {outfile}")

def process_fedprox(mu):
    clients, server, idx_clients = setupGC.setup_devices(splitedData, args)
    print("\nDone setting up FedProx devices.")

    print("Running FedProx ...")
    frame = run_fedprox(clients, server, args.num_rounds, args.local_epoch, mu, samp=None)
    if args.repeat is None:
        outfile = os.path.join(outpath, args.data_group, f'accuracy_fedprox_mu{mu}{suffix}.csv')
    else:
        outfile = os.path.join(outpath, args.data_group, "repeats", f'{args.repeat}_accuracy_fedprox_mu{mu}{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to: {outfile}")

def process_gcfl():
    clients, server, idx_clients = fl_setupGC.setup_devices(splitedData, args)
    print("\nDone setting up GCFL devices.")
    print("Running GCFL ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, args.data_group, f'accuracy_gcfl{suffix}.csv')
    else:
        outfile = os.path.join(outpath, args.data_group, "repeats", f'{args.repeat}_accuracy_gcfl{suffix}.csv')

    frame = run_cfl(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2)
    frame.to_csv(outfile)
    print(f"Wrote to: {outfile}")

def process_gcfl_dtw():
    clients, server, idx_clients = fl_setupGC.setup_devices(splitedData, args)
    print("\nDone setting up GCFLplus devices.")
    print("Running GCFL with DTW clustering ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, args.data_group, f'accuracy_gcfldtw{suffix}.csv')
    else:
        outfile = os.path.join(outpath, args.data_group, "repeats", f'{args.repeat}_accuracy_gcfldtw{suffix}.csv')

    frame = run_cfl_dtw(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2, args.standardize)
    frame.to_csv(outfile)
    print(f"Wrote to: {outfile}")

def process_gcfl_dtw_dWs():
    clients, server, idx_clients = fl_setupGC.setup_devices(splitedData, args)
    print("\nDone setting up GCFLplus devices.")
    print("Running GCFL with DTW clustering using dWs ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, args.data_group, f'accuracy_gcfldtwDWs{suffix}.csv')
    else:
        outfile = os.path.join(outpath, args.data_group, "repeats", f'{args.repeat}_accuracy_gcfldtwDWs{suffix}.csv')

    frame = run_cfl_dtw_dWs(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2, args.standardize)
    frame.to_csv(outfile)
    print(f"Wrote to: {outfile}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='CPU / GPU device.')
    parser.add_argument('--num_repeat', type=int, default=5,
                        help='number of repeating rounds to simulate;')
    parser.add_argument('--num_rounds', type=int, default=50,
                        help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=3,
                        help='number of local epochs;')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for inner solver;')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nlayer', type=int, default=2,
                        help='Number of GINconv layers')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for node classification.')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=123)

    parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
    parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default='small')

    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
    parser.add_argument('--overlap', help='whether clients have overlapped data',
                        type=bool, default=False)
    parser.add_argument('--standardize', help='whether to standardize the distance matrix',
                        type=bool, default=False)

    parser.add_argument('--datapath', help='the path for data',
                        type=str, default='./data')
    parser.add_argument('--outbase', help='the base path for outputs',
                        type=str, default='./outputs')

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    seed_dataSplit = 123

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.hidden = 64
    args.batch_size = 32
    args.lr = 0.001
    args.num_rounds = 200
    args.local_epoch = 3

    # for CFL
    if args.data_group == 'molecules':
        EPS_1 = 0.08
        EPS_2 = 0.3
    elif args.data_group == 'mix':
        EPS_1 = 0.06
        EPS_2 = 0.3
    elif args.data_group == 'biochem':
        EPS_1 = 0.05
        EPS_2 = 0.2

    # data input path and output path
    datapath = args.datapath
    outbase = args.outbase
    if args.overlap and args.standardize:
        outpath = os.path.join(outbase, f"standardizedDTW/multiDS-overlap")
    elif args.overlap:
        outpath = os.path.join(outbase, f"multiDS-overlap")
    elif args.standardize:
        outpath = os.path.join(outbase, f"standardizedDTW/multiDS-nonOverlap")
    else:
        outpath = os.path.join(outbase, f"multiDS-nonOverlap")
    Path(outpath).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {os.path.join(outpath, args.data_group)}")

    # preparing data
    if not args.convert_x:
        """ using original features """
        suffix = ""
        print("Preparing data (original features) ...")
    else:
        """ using node degree features """
        suffix = "_degrs"
        print("Preparing data (one-hot degree features) ...")
    Path(os.path.join(outpath, args.data_group)).mkdir(parents=True, exist_ok=True)

    if args.repeat is not None:
        Path(os.path.join(outpath, args.data_group, 'repeats')).mkdir(parents=True, exist_ok=True)

    splitedData, df_stats = setupGC.prepareData_multiDS(datapath, args.data_group, args.batch_size, convert_x=args.convert_x, seed=seed_dataSplit)
    print("Done")

    # save statistics of data on clients
    if args.repeat is None:
        outf = os.path.join(outpath, args.data_group, f'stats_trainData{suffix}.csv')
    else:
        outf = os.path.join(outpath, args.data_group, "repeats", f'{args.repeat}_stats_trainData{suffix}.csv')
    df_stats.to_csv(outf)
    print(f"Wrote to {outf}")

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # run
    process_selftrain(500)
    process_fedavg()
    process_fedprox(0.01)
    process_gcfl()
    process_gcfl_dtw()
    process_gcfl_dtw_dWs()
