import random
from random import choices
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

from models import GIN, serverGIN
from server import Server
from client import Client_GC
from utils import convert_to_nodeDegreeFeatures, get_stats, split_data, get_numGraphLabels


def prepareData_oneDS(datapath, data, batchSize, convert_x=False, seed=None):
    tudataset = TUDataset(f"{datapath}/TUDataset", data)
    if not tudataset[0].__contains__('x') or convert_x:
        new_graphs = convert_to_nodeDegreeFeatures(tudataset)
        # print("\t", data, len(new_graphs))
        graphs = new_graphs
    else:
        # print("\t", data, len(tudataset))
        graphs = [x for x in tudataset]

    graphs_tv, graphs_test = split_data(graphs, test=0.1, shuffle=True, seed=seed)

    dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)

    return graphs_tv, dataloader_test


def _randChunk(graphs_tv, num_client, overlap, seed=None):
    random.seed(seed)
    np.random.seed(seed)

    totalNum = len(graphs_tv)
    minSize = min(20, int(totalNum/num_client))
    graphs_tv_chunks = []
    if not overlap:
        for i in range(num_client):
            graphs_tv_chunks.append(graphs_tv[i*minSize:(i+1)*minSize])
        for g in graphs_tv[num_client*minSize:]:
            idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
            graphs_tv_chunks[idx_chunk].append(g)
    else:
        sizes = np.random.randint(low=20, high=50, size=num_client)
        for s in sizes:
            graphs_tv_chunks.append(choices(graphs_tv, k=s))
    return graphs_tv_chunks


def distributeData_oneDS(graphs_tv, dataloader_test, data, batchSize, num_client, seed=None, overlap=False):
    graphs_tv_chunks = _randChunk(graphs_tv, num_client, overlap, seed=seed)
    splitedData = {}
    df = pd.DataFrame()
    num_node_features = graphs_tv[0].num_node_features
    for idx, chunks in enumerate(graphs_tv_chunks):
        ds = f'{idx}-{data}'
        ds_tv = chunks
        ds_train, ds_val = train_test_split(ds_tv, train_size=0.9, test_size=0.1, shuffle=True, random_state=seed)
        # print(ds, f"train: {len(ds_train)}", f"val: {len(ds_val)}")
        dataloader_train = DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(ds_val, batch_size=batchSize, shuffle=True)
        num_graph_labels = get_numGraphLabels(ds_tv)
        splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                           num_node_features, num_graph_labels)
        df = get_stats(df, ds, ds_train, graphs_val=ds_val)
    # print(df)

    return splitedData, df


def prepareData_multiDS(datapath, group='mix', batchSize=32, convert_x=False, seed=None):
    assert group in ['molecules', 'mix', "biochem"]

    if group == 'molecules':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"]
    if group == 'mix':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",   # small molecules
                    "ENZYMES", "DD", "PROTEINS",                                # bioinformatics
                    "COLLAB", "IMDB-BINARY", "IMDB-MULTI"]                      # social networks
    if group == 'biochem':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                    "ENZYMES", "DD", "PROTEINS"]                               # bioinformatics

    splitedData = {}
    df = pd.DataFrame()
    for data in datasets:
        tudataset = TUDataset(f"{datapath}/TUDataset", data)
        if not tudataset[0].__contains__('x') or convert_x:
            new_graphs = convert_to_nodeDegreeFeatures(tudataset)
            graphs = new_graphs
        else:
            graphs = [x for x in tudataset]
        # print("\t", data, len(graphs))

        graphs_train, graphs_valtest = split_data(graphs, test=0.2, shuffle=True, seed=seed)
        graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)
        graphs_train, _ = split_data(graphs_train, train=50, shuffle=True, seed=seed)
        # print("  ", data, f"train: {len(graphs_train)}", f"val: {len(graphs_val)}", f"test: {len(graphs_test)}")

        num_node_features = graphs[0].num_node_features
        num_graph_labels = get_numGraphLabels(graphs_train)

        dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)

        splitedData[data] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                             num_node_features, num_graph_labels)

        df = get_stats(df, data, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)
    # print(df)

    return splitedData, df


def setup_devices(splitedData, args):
    idx_clients = {}
    clients = []
    i = 0
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloaders, num_node_features, num_graph_labels = splitedData[ds]
        cmodel_gc = GIN(num_node_features, args.hidden, num_graph_labels, args.nlayer)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel_gc.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        clients.append(Client_GC(cmodel_gc, idx, ds, dataloaders, optimizer, args))
        i += 1

    smodel = serverGIN(nlayer=args.nlayer, nhid=args.hidden)
    server = Server(smodel, args.device)
    return clients, server, idx_clients

