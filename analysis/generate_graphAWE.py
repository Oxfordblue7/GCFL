import os
import sys

import networkx as nx
import time

import pandas as pd
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

from AnonymousWalkKernel import GraphKernel
from AnonymousWalkKernel import AnonymousWalks

import multiprocessing as mp
from multiprocessing import Pool


def read_graphs(datapath, dataset):
    gk = GraphKernel()
    tudataset = TUDataset(datapath, dataset)
    for i, graph in enumerate(tudataset):
        g = to_networkx(graph, to_undirected=True)
        gk.read_graph_from_nx(g)
    # print(len(gk.graphs), nx.info(gk.graphs[0]))
    return gk


def generate_awe(g, klens=(3, 4, 5)):
    aw = AnonymousWalks(g)

    long_embeddings = []
    for klen in klens:
        # print(f"Anonymous walks of length {klen}:")
        # start = time.time()
        if klen == 3:
            embedding, meta = aw.embed(steps=klen, method='exact', keep_last=True, verbose=False)
        else:
            embedding, meta = aw.embed(steps=klen, method='sampling', keep_last=True, verbose=False, MC=100, delta=0.01, eps=0.1)
        # embedding, meta = aw.embed(steps=klen, method='sampling', keep_last=True, verbose=False, MC=100, delta=0.01, eps=0.1)
        # embedding, meta = aw.embed(steps=klen, method='exact', keep_last=True, verbose=False)
        # finish = time.time()

        # aws = meta['meta-paths']
        # print('Computed Embedding of {} dimension in {:.3f} sec.'.format(len(aws), finish - start))
        long_embeddings.extend(embedding)

    return long_embeddings


def generate_awe_v2(g, klens=(3, 4, 5)):
    aw = AnonymousWalks(g)

    long_embeddings = []
    for klen in klens:
        # print(f"Anonymous walks of length {klen}:")
        # start = time.time()
        embedding, meta = aw.embed(steps=klen, method='sampling', keep_last=True, verbose=False, MC=100, delta=0.01, eps=0.1)
        # finish = time.time()

        # aws = meta['meta-paths']
        # print('Computed Embedding of {} dimension in {:.3f} sec.'.format(len(aws), finish - start))
        long_embeddings.extend(embedding)

    return long_embeddings


def runner(ds):
    datapath = './data'
    outpath = './outputs/AWEs'
    gk = read_graphs(datapath, ds)
    klens = (3, 4, 5)

    df = pd.DataFrame()
    print("Dataset: {}; #graphs: {}".format(ds, len(gk.graphs)))
    for i, g in enumerate(gk.graphs):
        # if (i+1)%10 == 0:
        print(f"  > {i+1}th graph")
        embs = generate_awe(g, klens)
        df[i] = embs

    print(df)
    df.to_csv(os.path.join(outpath, f'AWEs_{ds}_{klens[0]}-{klens[-1]}_sampling.csv'), index=False)
    print("Wrote to file:", os.path.join(outpath, f'AWEs_{ds}_{klens[0]}-{klens[-1]}_sampling.csv'))


def runner2(graphs):
    ds = "COLLAB"
    gk = GraphKernel()
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gk.read_graph_from_nx(g)

    outpath = './outputs/AWEs'
    klens = (3, 4, 5)

    df = pd.DataFrame()
    print("Dataset: {}-{}; #graphs: {}".format(s, ds, len(gk.graphs)))
    for i, g in enumerate(gk.graphs):
        # if (i+1)%10 == 0:
        print(f"  > {i + 1}th graph")
        embs = generate_awe(g, klens)
        df[i] = embs

    print(df)
    df.to_csv(os.path.join(outpath, f'{s}-AWEs_{ds}_{klens[0]}-{klens[-1]}_sampling.csv'), index=False)
    print("Wrote to file:", os.path.join(outpath, f'{s}-AWEs_{ds}_{klens[0]}-{klens[-1]}_sampling.csv'))


if __name__ == "__main__":
    # datasets = [#"MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
    #             # "ENZYMES",
    #             # "DD", "PROTEINS",  # bioinformatics
    #             "COLLAB", "IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY"]  # social networks
    #
    # with Pool(4) as p:
    #     p.map(runner, datasets)
    # runner("MUTAG")
    # runner(sys.argv[1])

    tudataset = TUDataset("./data", "COLLAB")
    graphs = [x for x in tudataset]
    parts = []
    for s in range(5):
        parts.append(graphs[s * 1000:(s + 1) * 1000])

    runner2(parts[0])

    # with Pool(5) as p:
    #     # p.map(runner2, [0, 1, 2, 3, 4])
    #     p.map(runner2, parts)

