import os
import argparse
import random
import copy

import torch
from pathlib import Path

import setupGC
from training import *


def process_selftrain(clients, server, local_epoch):
    print("Self-training ...")
    df = pd.DataFrame()
    allAccs = run_selftrain_GC(clients, server, local_epoch)
    for k, v in allAccs.items():
        df.loc[k, [f'train_acc', f'val_acc', f'test_acc']] = v
    print(df)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_selftrain_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_selftrain_GC{suffix}.csv')
    df.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_fedavg(clients, server):
    print("\nDone setting up FedAvg devices.")

    print("Running FedAvg ...")
    frame = run_fedavg(clients, server, args.num_rounds, args.local_epoch, samp=None)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedavg_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedavg_GC{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_fedprox(clients, server, mu):
    print("\nDone setting up FedProx devices.")

    print("Running FedProx ...")
    frame = run_fedprox(clients, server, args.num_rounds, args.local_epoch, mu, samp=None)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedprox_mu{mu}_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedprox_mu{mu}_GC{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_gcfl(clients, server):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcfl_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcfl_GC{suffix}.csv')

    frame = run_gcfl(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_gcflplus(clients, server):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL plus ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcflplus_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcflplus_GC{suffix}.csv')

    frame = run_gcflplus(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2, args.seq_length, args.standardize)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_gcflplusdWs(clients, server):
    print("\nDone setting up CFL devices.")
    print("Running GCFL plus dWs ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcflplusDWs_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcflplusDWs_GC{suffix}.csv')

    frame = run_gcflplus_dWs(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2, args.seq_length, args.standardize)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='CPU / GPU device.')
    parser.add_argument('--num_repeat', type=int, default=5,
                        help='number of repeating rounds to simulate;')
    parser.add_argument('--num_rounds', type=int, default=200,
                        help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for inner solver;')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of GINconv layers')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for node classification.')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=123)

    parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
    parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default='mix')

    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
    parser.add_argument('--overlap', help='whether clients have overlapped data',
                        type=bool, default=False)
    parser.add_argument('--standardize', help='whether to standardize the distance matrix',
                        type=bool, default=False)
    parser.add_argument('--seq_length', help='the length of the gradient norm sequence',
                        type=int, default=10)
    parser.add_argument('--epsilon1', help='the threshold epsilon1 for GCFL',
                        type=float, default=0.01)
    parser.add_argument('--epsilon2', help='the threshold epsilon2 for GCFL',
                        type=float, default=0.1)

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    seed_dataSplit = 123

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    EPS_1 = args.epsilon1
    EPS_2 = args.epsilon2

    # TODO: change the data input path and output path
    datapath = "/local/scratch/hxie45/scripts/GCFL/data/"
    outbase = f'/local/scratch/hxie45/scripts/GCFL/outputs/seqLen{args.seq_length}'

    if args.overlap and args.standardize:
        outpath = os.path.join(outbase, f"standardizedDTW/multiDS-overlap")
    elif args.overlap:
        outpath = os.path.join(outbase, f"multiDS-overlap")
    elif args.standardize:
        outpath = os.path.join(outbase, f"standardizedDTW/multiDS-nonOverlap")
    else:
        outpath = os.path.join(outbase, f"multiDS-nonOverlap")
    outpath = os.path.join(outpath, args.data_group, f'eps_{EPS_1}_{EPS_2}')
    Path(outpath).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {outpath}")

    # preparing data
    if not args.convert_x:
        """ using original features """
        suffix = ""
        print("Preparing data (original features) ...")
    else:
        """ using node degree features """
        suffix = "_degrs"
        print("Preparing data (one-hot degree features) ...")

    if args.repeat is not None:
        Path(os.path.join(outpath, 'repeats')).mkdir(parents=True, exist_ok=True)

    splitedData, df_stats = setupGC.prepareData_multiDS(datapath, args.data_group, args.batch_size, convert_x=args.convert_x, seed=seed_dataSplit)
    print("Done")

    # save statistics of data on clients
    if args.repeat is None:
        outf = os.path.join(outpath, f'stats_trainData{suffix}.csv')
    else:
        outf = os.path.join(outpath, "repeats", f'{args.repeat}_stats_trainData{suffix}.csv')
    df_stats.to_csv(outf)
    print(f"Wrote to {outf}")

    init_clients, init_server, init_idx_clients = setupGC.setup_devices(splitedData, args)
    print("\nDone setting up devices.")

    process_selftrain(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), local_epoch=100)
    process_fedavg(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))
    process_fedprox(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), mu=0.01)
    process_gcfl(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))
    process_gcflplus(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))
    process_gcflplusdWs(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))
