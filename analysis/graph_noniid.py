import os
import pickle
import json
import random

import torch
import numpy as np
import pandas as pd
import seaborn as sns
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx, degree
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.spatial import distance
from ..utils import split_data, use_node_attributes, convert_to_nodeDegreeFeatures
from multiprocessing import Pool
from ..setupGC import _randChunk
from AnonymousWalkKernel import GraphKernel
from generate_graphAWE import generate_awe_v2



def main_ratio_KStest_distDiff_nodeDegrees(ds_pair):
    ds1, ds2 = ds_pair
    print(ds_pair)
    dists_diff1 = dict_dist_diff_degrs[ds1]
    dists_diff2 = dict_dist_diff_degrs[ds2]
    # ks_pvalues = np.zeros((len(dists_diff1), len(dists_diff2)))
    count_noniid = 0
    for i in range(len(dists_diff1)):
        for j in range(len(dists_diff2)):
            # ks_pvalues[i, j] = stats.ks_2samp(dists_diff1[i], dists_diff2[j])[1]
            if stats.ks_2samp(dists_diff1[i], dists_diff2[j])[1] <= 0.01:
                count_noniid += 1
    # ratio = len(ks_pvalues[ks_pvalues <= 0.01]) * 1. / (len(dists_diff1) * len(dists_diff2))
    ratio = count_noniid * 1. / (len(dists_diff1) * len(dists_diff2))
    df = pd.DataFrame()
    df.loc[ds1, ds2] = ratio
    # df.to_csv(f"./outputs/remote/featureStats/tmps/tmp_ratio_noniid_nodeDegrees_{ds1}_{ds2}.csv")
    df.to_csv(f"./outputs/tmp_ratio_noniid_nodeDegrees_{ds1}_{ds2}.csv")


def aggregate_ratioFiles(ks, indir, suffix):
    # ks = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",
    #       "ENZYMES", "DD", "PROTEINS",
    #       "IMDB-BINARY", "IMDB-MULTI", "COLLAB", "REDDIT-BINARY"]
    # ks = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "MUTAG", "NCI1", "PROTEINS", "PTC_MR"]
    df = pd.DataFrame(index=ks, columns=ks)
    for tmpfile in os.listdir(os.path.join(indir, 'tmps')):
        if tmpfile.startswith(f'tmp_ratio_noniid_{suffix}_'):
            df_tmp = pd.read_csv(os.path.join(indir, "tmps", tmpfile), header=0, index_col=0)
            df.loc[df_tmp.columns, df_tmp.index] = df_tmp.values
            df.loc[df_tmp.index, df_tmp.columns] = df_tmp.values
    pd.options.display.max_columns = None
    print(df)
    df.to_csv(os.path.join(indir, f'ratio_pvalues_KStest_{suffix}.csv'))



def _get_client_trainData_oneDS(tudataset, num_client, overlap, seed_dataSplit):
    # graphs = [(idx, x) for idx, x in enumerate(tudataset)]
    indices = list(range(len(tudataset)))

    client_trainIndices = []
    y = torch.cat([graph.y for graph in tudataset])
    indices_tv, _ = train_test_split(indices, test_size=0.1, stratify=y, shuffle=True, random_state=seed_dataSplit)
    # indices_tv, _ = split_data(indices, test=0.1, shuffle=True, seed=seed_dataSplit)
    indices_tv_chunks = _randChunk(indices_tv, num_client, overlap, seed=seed_dataSplit)
    for idx, chunks in enumerate(indices_tv_chunks):
        indices_train, _ = train_test_split(chunks, train_size=0.9, test_size=0.1, shuffle=True, random_state=seed_dataSplit)
        client_trainIndices.append(indices_train)

    return client_trainIndices #[[idx, ...], [...], ...]



def _get_avg_JSdist_awe_byClient(df_awe, client1, client2):
    jsDists = []
    for idx1 in client1:
        for idx2 in client2:
            jsDists.append(distance.jensenshannon(df_awe[str(idx1)], df_awe[str(idx2)]))
    return np.nanmean(jsDists), np.nanstd(jsDists)



def _generate_distribution_nodeLabelSimilarity(tudataset):
    dict_sim = {}
    for idx, g in enumerate(tudataset):
        sims = []
        for idx_e in range(len(g.edge_index[0])):
            n1 = g.edge_index[0][idx_e]
            n2 = g.edge_index[1][idx_e]
            sim = 1 - distance.cosine(g.x[n1], g.x[n2])  # 0 or 1
            sims.append(sim)
        distribution_sim = []
        for v in sorted(set(sims)):
            distribution_sim.append(sims.count(v))
        dict_sim[idx] = distribution_sim
    return dict_sim



def _get_avg_JSdist_simDistribution_byClient(dict_sim, client1, client2):
    jsDist = []
    for idx1 in client1:
        for idx2 in client2:
            jsDist.append(distance.jensenshannon(dict_sim[idx1], dict_sim[idx2]))
    return np.nanmean(jsDist), np.nanstd(jsDist)


def _pca_analysis(client_names, df_awe, client_indices):
    df = pd.DataFrame(columns=client_names)
    for i in range(len(client_names)):
        df[client_names[i]] = list(df_awe[[str(x) for x in client_indices[i]]].mean(axis=1))
    # print(df)
    df = df.T
    x = df.values

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principal = pca.fit_transform(x)
    df_pca = pd.DataFrame(data=principal, index=client_names, columns=['pc1', 'pc2'])

    return df_pca

def _generate_graphAWE_proteins(ds, graphs, indices):
    outpath = './outputs/AWEs'
    gk = GraphKernel()
    for graph in graphs:
        g = to_networkx(graph, to_undirected=True)
        gk.read_graph_from_nx(g)

    klens = (3, 4, 5, 6, 7)
    df = pd.DataFrame()
    print("Dataset: {}; #graphs: {}".format(ds, len(gk.graphs)))
    for i in range(len(indices)):
        embs = generate_awe_v2(gk.graphs[i], klens)
        df[indices[i]] = embs
    # print(df)
    df.to_csv(os.path.join(outpath, f'AWEs_{ds}_{klens[0]}-{klens[-1]}.csv'), index=False)
    print("Wrote to file:", os.path.join(outpath, f'AWEs_{ds}_{klens[0]}-{klens[-1]}.csv'))
    # df.to_csv(os.path.join(outpath, f'AWEs_{ds}_{klens[0]}-{klens[-1]}_sampling.csv'), index=False)
    # print("Wrote to file:", os.path.join(outpath, f'AWEs_{ds}_{klens[0]}-{klens[-1]}_sampling.csv'))
    return df

def analyze_struct_feature_proteins():
    seed_dataSplit = 123
    datapath = "./data/"
    dataset = 'PROTEINS'
    num_client = 30
    overlap = False

    tudataset = TUDataset(f"{datapath}/TUDataset", dataset)
    client_trainIndices = _get_client_trainData_oneDS(tudataset, num_client, overlap, seed_dataSplit)

    client_names = [f'{i}-{dataset}' for i in range(num_client)]

    """ structral similarity """
    df_awe = pd.read_csv(f"./outputs/AWEs/AWEs_{dataset}_3-7.csv", index_col=None, header=0)

    jsDist = np.zeros((num_client, num_client))
    jsDist_std = np.zeros((num_client, num_client))
    for i in range(num_client):
        for j in range(i+1, num_client):
            diff, diff_std = _get_avg_JSdist_awe_byClient(df_awe, client_trainIndices[i], client_trainIndices[j])
            print((i, j), diff, diff_std)
            jsDist[i, j] = diff
            jsDist[j, i] = diff
            jsDist_std[i, j] = diff_std
            jsDist_std[j, i] = diff_std

    df = pd.DataFrame(jsDist, index=client_names, columns=client_names)
    # print(df)
    df.to_csv(f"./outputs//heteroAnalysis/{dataset}/jsDists_awes_{overlap}-{num_client}clients.csv", header=True, index=True)
    df = pd.DataFrame(jsDist_std, index=client_names, columns=client_names)
    df.to_csv(f"./outputs/heteroAnalysis/{dataset}/std_jsDists_awes_{overlap}-{num_client}clients.csv",
        header=True, index=True)

    """ feature similarity """
    dict_sim = _generate_distribution_nodeLabelSimilarity(tudataset)

    df_js = pd.DataFrame(0, index=client_names, columns=client_names)
    df_js_std = pd.DataFrame(0, index=client_names, columns=client_names)
    for i in range(num_client):
        for j in range(i + 1, num_client):
            diff, diff_std = _get_avg_JSdist_simDistribution_byClient(dict_sim, client_trainIndices[i], client_trainIndices[j])
            print((i, j), diff, diff_std)
            df_js.loc[client_names[i], client_names[j]] = diff
            df_js.loc[client_names[j], client_names[i]] = diff
            df_js_std.loc[client_names[i], client_names[j]] = diff_std
            df_js_std.loc[client_names[j], client_names[i]] = diff_std

    df_js.to_csv(f"./outputs/heteroAnalysis/{dataset}/jsDists_nodeLabelSim_{overlap}-{num_client}clients.csv",
                 header=True, index=True)
    df_js_std.to_csv(f"./outputs/heteroAnalysis/{dataset}/std_jsDists_nodeLabelSim_{overlap}-{num_client}clients.csv",
                     header=True, index=True)

# cluster map
def _clustermap(df, plotfile):
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.clustermap(df, cmap='rocket_r')
    # plt.show()
    plt.savefig(plotfile)



def _get_client_trainData_multiDS_mixtiny(list_tudatasets, seed_dataSplit):
    client_trainIndices = []
    for tudataset in list_tudatasets:
        graphs = [(idx, x) for idx, x in enumerate(tudataset)]
        y = torch.cat([graph.y for graph in tudataset])
        graphs_train, _ = train_test_split(graphs, test_size=0.2, stratify=y, shuffle=True, random_state=seed_dataSplit)
        y2 = torch.cat([x[1].y for x in graphs_train])
        graphs_train, _ = train_test_split(graphs_train, train_size=50, stratify=y2, shuffle=True,
                                               random_state=seed_dataSplit)
        indices_train = [x[0] for x in graphs_train]
        client_trainIndices.append(indices_train)
        # if ds == "COLLAB":
        #     indices_train_collab = indices_train
    return client_trainIndices#, indices_train_collab

def _generate_graphAWE_mixtiny(ds, graphs, indices):
    outpath = './outputs/AWEs/mixtiny'
    gk = GraphKernel()
    for graph in graphs:
        g = to_networkx(graph, to_undirected=True)
        gk.read_graph_from_nx(g)

    klens = (3, 4, 5, 6, 7)
    df = pd.DataFrame()
    print("Dataset: {}; #graphs: {}".format(ds, len(gk.graphs)))
    for i in range(len(indices)):
        embs = generate_awe_v2(gk.graphs[i], klens)
        df[indices[i]] = embs
    # print(df)
    df.to_csv(os.path.join(outpath, f'AWEs_{ds}_{klens[0]}-{klens[-1]}.csv'), index=False)
    print("Wrote to file:", os.path.join(outpath, f'AWEs_{ds}_{klens[0]}-{klens[-1]}.csv'))
    # df.to_csv(os.path.join(outpath, f'AWEs_{ds}_{klens[0]}-{klens[-1]}_sampling.csv'), index=False)
    # print("Wrote to file:", os.path.join(outpath, f'AWEs_{ds}_{klens[0]}-{klens[-1]}_sampling.csv'))
    return df



def _get_avg_JSdist_awe_byClient_mixtiny(df_awe1, client_trainIndices1, df_awe2, client_trainIndices2):
    jsDist = []
    for idx1 in client_trainIndices1:
        for idx2 in client_trainIndices2:
            jsDist.append(distance.jensenshannon(df_awe1[str(idx1)], df_awe2[str(idx2)]))
    # print(np.mean(cosSims), max(cosSims), min(cosSims), cosSims[:20])
    return np.mean(jsDist), np.std(jsDist)


def _generate_distribution_nodeLabelSimilarity_mixtiny(graphs, clientIndices):
    dict_sim = {}
    for i in range(len(graphs)):
        g = graphs[i]
        # distribution_sim = []
        sims = []
        for idx_e in range(len(g.edge_index[0])):
            n1 = g.edge_index[0][idx_e]
            n2 = g.edge_index[1][idx_e]
            sim = 1 - distance.cosine(g.x[n1], g.x[n2]) # 0 or 1
            sims.append(sim)
        distribution_sim = []
        for v in sorted(set(sims)):
            distribution_sim.append(sims.count(v))
        dict_sim[clientIndices[i]] = distribution_sim
    return dict_sim

def _get_avg_JSdist_simDistribution_byClient_mixtiny(dict_sim1, dict_sim2):
    jsDist = []
    for v1 in dict_sim1.values():
        for v2 in dict_sim2.values():
            jsDist.append(distance.jensenshannon(v1, v2))
    return np.mean(jsDist), np.std(jsDist)


def _structural_analysis_mixtiny_JS(datasets, dfs_awe, client_trainIndices):
    df_js = pd.DataFrame(0, index=datasets, columns=datasets)
    df_js_std = pd.DataFrame(0, index=datasets, columns=datasets)
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            diff, diff_std = _get_avg_JSdist_awe_byClient_mixtiny(dfs_awe[i], client_trainIndices[i], dfs_awe[j],
                                                                  client_trainIndices[j])
            print((i, j), diff, diff_std)
            df_js.loc[datasets[i], datasets[j]] = diff
            df_js.loc[datasets[j], datasets[i]] = diff
            df_js_std.loc[datasets[i], datasets[j]] = diff_std
            df_js_std.loc[datasets[j], datasets[i]] = diff_std

    df_js.to_csv(f"./outputs/heteroAnalysis/mixtiny/jsDists_awes.csv",
                     header=True, index=True)
    df_js_std.to_csv(f"./outputs/heteroAnalysis/mixtiny/std_jsDists_awes.csv",
                         header=True, index=True)


def _feature_analysis_mixtiny_JS(datasets, all_dict_sim):
    df_js = pd.DataFrame(0, index=datasets, columns=datasets)
    df_js_std = pd.DataFrame(0, index=datasets, columns=datasets)
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            diff, diff_std = _get_avg_JSdist_simDistribution_byClient_mixtiny(all_dict_sim[i], all_dict_sim[j])
            print((i, j), diff, diff_std)
            df_js.loc[datasets[i], datasets[j]] = diff
            df_js.loc[datasets[j], datasets[i]] = diff
            df_js_std.loc[datasets[i], datasets[j]] = diff_std
            df_js_std.loc[datasets[j], datasets[i]] = diff_std

    df_js.to_csv(f"./outputs/heteroAnalysis/mixtiny/jsDists_nodeLabelSim.csv",
                 header=True, index=True)
    df_js_std.to_csv(f"./outputs/heteroAnalysis/mixtiny/std_jsDists_nodeLabelSim.csv",
                     header=True, index=True)


def analyze_struct_feature_mixtiny():
    seed_dataSplit = 123
    datapath = "./data/"
    datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                "ENZYMES", "DD", "PROTEINS",  # bioinformatics
                "COLLAB", "IMDB-BINARY", "IMDB-MULTI"]  # social networks

    list_tudatasets = []
    for ds in datasets:
        tudataset = TUDataset(f"{datapath}/TUDataset", ds)
        if not tudataset[0].__contains__('x'):
            list_tudatasets.append(convert_to_nodeDegreeFeatures(tudataset))
        else:
            list_tudatasets.append([x for x in tudataset])
    # list_tudatasets = [TUDataset(f"{datapath}/TUDataset", ds) for ds in datasets]
    client_trainIndices = _get_client_trainData_multiDS_mixtiny(list_tudatasets, seed_dataSplit)

    """ structrual analysis """
    dfs_awe = []
    for i in range(len(datasets)):
        df_awe = pd.read_csv(f"./outputs/AWEs/mixtiny/AWEs_{datasets[i]}_3-7.csv",
                             index_col=None, header=0)
        dfs_awe.append(df_awe)
    _structural_analysis_mixtiny_JS(datasets, dfs_awe, client_trainIndices)

    """ feature analysis """
    all_dict_sim = []
    for i in range(len(datasets)):
        graphs = [list_tudatasets[i][idx] for idx in client_trainIndices[i]]
        all_dict_sim.append(_generate_distribution_nodeLabelSimilarity_mixtiny(graphs, client_trainIndices[i]))

    _feature_analysis_mixtiny_JS(datasets, all_dict_sim)



def _get_client_trainData_multiDS_mix(list_tudatasets, seed_dataSplit):
    client_trainIndices = {}
    for ds, tudataset in list_tudatasets.items():
        graphs = [(idx, x) for idx, x in enumerate(tudataset)]
        y = torch.cat([graph.y for graph in tudataset])
        graphs_train, _ = train_test_split(graphs, test_size=0.2, stratify=y, shuffle=True, random_state=seed_dataSplit)
        indices_train = [x[0] for x in graphs_train]
        client_trainIndices[ds] = indices_train
    return client_trainIndices

def _generate_graphAWE_mix(ds, graphs, indices):
    outpath = './outputs/AWEs/mix'
    gk = GraphKernel()
    for graph in graphs:
        g = to_networkx(graph, to_undirected=True)
        gk.read_graph_from_nx(g)

    klens = (3, 4, 5, 6, 7)
    df = pd.DataFrame()
    print("Dataset: {}; #graphs: {}".format(ds, len(gk.graphs)))
    for i in range(len(indices)):
        embs = generate_awe_v2(gk.graphs[i], klens)
        df[str(indices[i])] = embs
    # print(df)
    df.to_csv(os.path.join(outpath, f'AWEs_{ds}_{klens[0]}-{klens[-1]}.csv'), index=False)
    print("Wrote to file:", os.path.join(outpath, f'AWEs_{ds}_{klens[0]}-{klens[-1]}.csv'))
    return df

def _get_avg_cosSim_byClient_mix(df_awe1, client_trainIndices1, df_awe2, client_trainIndices2):
    cosSims = []
    for idx1 in client_trainIndices1:
        for idx2 in client_trainIndices2:
            cosSims.append(1 - distance.cosine(df_awe1[str(idx1)], df_awe2[str(idx2)]))
    # print(np.mean(cosSims), max(cosSims), min(cosSims), cosSims[:20])
    return np.mean(cosSims), np.std(cosSims)


def _structural_analysis_mix(datasets, dfs_awe, client_trainIndices, suffix):
    df_cosSim = pd.DataFrame(1, index=datasets, columns=datasets)
    df_cosSim_std = pd.DataFrame(0, index=datasets, columns=datasets)
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            ds1 = datasets[i]
            ds2 = datasets[j]
            sim, sim_std = _get_avg_cosSim_byClient_mix(dfs_awe[ds1], client_trainIndices[ds1], dfs_awe[ds2],
                                                            client_trainIndices[ds2])
            print((i, j), sim, sim_std)
            df_cosSim.loc[datasets[i], datasets[j]] = sim
            df_cosSim.loc[datasets[j], datasets[i]] = sim
            df_cosSim_std.loc[datasets[i], datasets[j]] = sim_std
            df_cosSim_std.loc[datasets[j], datasets[i]] = sim_std

    df_cosSim.to_csv(f"./outputs/heteroAnalysis/mix/cosSims_awes{suffix}.csv",
                     header=True, index=True)
    df_cosSim_std.to_csv(f"./outputs/heteroAnalysis/mix/std_cosSims{suffix}.csv",
                         header=True, index=True)

def _generate_distribution_nodeFeatureDifference_mix(graphs, clientIndices):
    dict_diff = {}
    for i in range(len(graphs)):
        g = graphs[i]
        distribution_diff = []
        vec = torch.nonzero((g.x == 1), as_tuple=True)[1]
        for idx_e in range(len(g.edge_index[0])):
            n1 = g.edge_index[0][idx_e]
            n2 = g.edge_index[1][idx_e]
            diff = abs(vec[n1] - vec[n2])
            distribution_diff.append(diff.item())
        dict_diff[clientIndices[i]] = distribution_diff
    return dict_diff

def _get_ratio_identicalDistribution_byClient_mix(dict_diff1, dict_diff2):
    ksstats = []
    count = 0
    for vec1 in dict_diff1.values():
        for vec2 in dict_diff2.values():
            res = stats.ks_2samp(vec1, vec2)
            if res[1] > 0.1:
                count += 1
            ksstats.append(res[0])
    ratio = count * 1. / (len(dict_diff1) * len(dict_diff2))
    return ratio, np.mean(ksstats)

def _feature_analysis_mix(datasets, all_dict_diff, suffix):
    df_ratios = pd.DataFrame(1, index=datasets, columns=datasets)
    df_ksstats = pd.DataFrame(0, index=datasets, columns=datasets)
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            ds1 = datasets[i]
            ds2 = datasets[j]
            ratio, ks = _get_ratio_identicalDistribution_byClient_mix(all_dict_diff[ds1], all_dict_diff[ds2])
            print((i, j), ratio)
            df_ratios.loc[datasets[i], datasets[j]] = ratio
            df_ratios.loc[datasets[j], datasets[i]] = ratio
            df_ksstats.loc[datasets[i], datasets[j]] = ks
            df_ksstats.loc[datasets[j], datasets[i]] = ks

    df_ratios.to_csv(f"./outputs/heteroAnalysis/mix/ratiosIID_nodeLabelDiff{suffix}.csv",
                     header=True, index=True)
    df_ksstats.to_csv(f"./outputs/heteroAnalysis/mix/ksStats_nodeLabelDiff{suffix}.csv",
                      header=True, index=True)

def compare_struct_feature_mix():
    seed_dataSplit = 123
    datapath = "./data/"
    datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                "ENZYMES", "DD", "PROTEINS",  # bioinformatics
                "COLLAB", "IMDB-BINARY", "IMDB-MULTI"]  # social networks

    clusters_cfl = [['MUTAG', 'BZR', 'AIDS', 'ENZYMES', 'PROTEINS', 'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI'],
                    ['COX2', 'DHFR', 'PTC_MR', 'NCI1', 'DD']]

    clusters_cfldtw = [['MUTAG', 'BZR', 'DHFR', 'PTC_MR', 'NCI1', 'ENZYMES', 'DD', 'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI'],
                       ['COX2', 'AIDS', 'PROTEINS']]

    list_tudatasets = {}
    for ds in datasets:
        tudataset = TUDataset(f"{datapath}/TUDataset", ds)
        if not tudataset[0].__contains__('x'):
            list_tudatasets[ds] = convert_to_nodeDegreeFeatures(tudataset)
        else:
            list_tudatasets[ds] = [x for x in tudataset]
    client_trainIndices = _get_client_trainData_multiDS_mix(list_tudatasets, seed_dataSplit)

    """ structrual analysis """
    dfs_awe = {}
    for ds in datasets:
        df_awe = pd.read_csv(os.path.join('./outputs/AWEs/mix',
                                          f'AWEs_{ds}_3-7.csv'), header=0, index_col=None)
        dfs_awe[ds] = df_awe

    for i, c in enumerate(clusters_cfl):
        _structural_analysis_mix(c, dfs_awe, client_trainIndices, f'_cfl_cluster{i}')
    for i, c in enumerate(clusters_cfldtw):
        _structural_analysis_mix(c, dfs_awe, client_trainIndices, f'_cfldtw_cluster{i}')

    """ feature analysis """
    all_dict_diff = {}
    for ds in datasets:
        graphs = [list_tudatasets[ds][idx] for idx in client_trainIndices[ds]]
        all_dict_diff[ds] = _generate_distribution_nodeFeatureDifference_mix(graphs, client_trainIndices[ds])

    for i, c in enumerate(clusters_cfl):
        _feature_analysis_mix(c, all_dict_diff, f'_cfl_cluster{i}')
    for i, c in enumerate(clusters_cfldtw):
        _feature_analysis_mix(c, all_dict_diff, f'_cfldtw_cluster{i}')


def calc_similarity_clusterwise_mix():
    dataset = 'mix'
    inpath = f'./outputs/heteroAnalysis/{dataset}'

    df_eval = pd.DataFrame()
    for algo in ['cfl', 'cfldtw']:
        for i in range(2):
            suffix = f'_{algo}_cluster{i}'
            df_structSim = pd.read_csv(os.path.join(inpath, f'cosSims_awes{suffix}.csv'), header=0, index_col=0)
            df_ksStats = pd.read_csv(os.path.join(inpath, f'ksStats_nodeLabelDiff{suffix}.csv'), header=0, index_col=0)
            df_IIDratios = pd.read_csv(os.path.join(inpath, f'ratiosIID_nodeLabelDiff{suffix}.csv'), header=0, index_col=0)

            c1 = df_structSim.values.mean()
            c2 = df_ksStats.values.mean()
            c3 = df_IIDratios.values.mean()
            df_eval.loc[f'{algo}_cluster{i}', 'mean_pairwise_structural_similarity'] = c1
            df_eval.loc[f'{algo}_cluster{i}', 'mean_pairwise_feature_similarity_ksStats'] = c2
            df_eval.loc[f'{algo}_cluster{i}', 'mean_pairwise_feature_similarity_IIDratios'] = c3
            df_eval.loc[f'{algo}_cluster{i}', 'similarity+ratio-ksStats'] = c1 + c3 - c2

            _clustermap(df_structSim, os.path.join(inpath, f"clustermap_cosSims_awes{suffix}.pdf"))
            _clustermap(df_ksStats, os.path.join(inpath, f"clustermap_ksStats_nodeLabelDiff{suffix}.pdf"))
            _clustermap(df_IIDratios, os.path.join(inpath, f"clustermap_ratiosIID_nodeLabelDiff{suffix}.pdf"))
            df_overall = df_structSim + df_IIDratios - df_ksStats
            _clustermap(df_overall, os.path.join(inpath, f"clustermap_strucFeatSimilarity_nodeLabelDiff{suffix}.pdf"))

    df_eval.to_csv(os.path.join(inpath, f'comparison_mean_pairwise_similarity_byClusters.csv'), header=True, index=True)


def _get_avg_JSdist_awe_byClient_mix(df_awe1, df_awe2):
    jsDist = []
    for c1 in df_awe1.columns:
        for c2 in df_awe2.columns:
            jsDist.append(distance.jensenshannon(df_awe1[c1], df_awe2[c2]))
    # print(np.mean(cosSims), max(cosSims), min(cosSims), cosSims[:20])
    return np.nanmean(jsDist), np.nanstd(jsDist)

def _get_avg_JSdist_simDistribution_byClient_mix(dict_sim1, dict_sim2):
    jsDist = []
    for v1 in dict_sim1.values():
        for v2 in dict_sim2.values():
            jsDist.append(distance.jensenshannon(v1, v2))
    return np.nanmean(jsDist), np.nanstd(jsDist)

def _generate_distribution_nodeLabelSimilarity_mix(graphs):
    dict_sim = {}
    for i in range(len(graphs)):
        g = graphs[i]
        # distribution_sim = []
        sims = []
        for idx_e in range(len(g.edge_index[0])):
            n1 = g.edge_index[0][idx_e]
            n2 = g.edge_index[1][idx_e]
            sim = 1 - distance.cosine(g.x[n1], g.x[n2]) # 0 or 1
            sims.append(sim)
        distribution_sim = []
        for v in sorted(set(sims)):
            distribution_sim.append(sims.count(v))
        dict_sim[i] = distribution_sim
    return dict_sim

def _get_avg_JSdist_awe_byClient_one(df_awe, cols1, cols2):
    jsDists = []
    for c1 in cols1:
        for c2 in cols2:
            jsDists.append(distance.jensenshannon(df_awe[c1], df_awe[c2]))
    # print(np.mean(jsDists), max(jsDists), min(jsDists), jsDists[:20])
    # if i == 1 and j == 3:
    #     print(np.nanmean(jsDists), max(jsDists), min(jsDists))
    return np.nanmean(jsDists), np.nanstd(jsDists)

def _get_avg_JSdist_simDistribution_byClient_one(dict_sim, client1, client2):
    jsDist = []
    for idx1 in client1:
        for idx2 in client2:
            jsDist.append(distance.jensenshannon(dict_sim[int(idx1)], dict_sim[int(idx2)]))
    return np.nanmean(jsDist), np.nanstd(jsDist)

def analyze_struct_feature_heterogeneity():
    """ directly compare structure & feature heterogeneity """
    datapath = "./data/"

    # ######################## cross datasets ##########################
    # dss = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",
    #         "ENZYMES", "DD", "PROTEINS",
    #         "IMDB-BINARY", "IMDB-MULTI", "COLLAB", "REDDIT-BINARY"]
    datasets = ["ENZYMES", "DD", "PROTEINS"]
    suffix = 'bioinfo'
    list_tudatasets = []
    for ds in datasets:
        tudataset = TUDataset(f"{datapath}/TUDataset", ds)
        if not tudataset[0].__contains__('x'):
            list_tudatasets.append(convert_to_nodeDegreeFeatures(tudataset))
        else:
            list_tudatasets.append([x for x in tudataset])

    """ structrual analysis """
    dfs_awe = []
    for i in range(len(datasets)):
        df_awe = pd.read_csv(f"./outputs/AWEs/mix/AWEs_{datasets[i]}_3-7.csv",
                             index_col=None, header=0)
        dfs_awe.append(df_awe)

    df_js = pd.DataFrame(0, index=datasets, columns=datasets)
    df_js_std = pd.DataFrame(0, index=datasets, columns=datasets)
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            diff, diff_std = _get_avg_JSdist_awe_byClient_mix(dfs_awe[i], dfs_awe[j])
            print((i, j), diff, diff_std)
            df_js.loc[datasets[i], datasets[j]] = diff
            df_js.loc[datasets[j], datasets[i]] = diff
            df_js_std.loc[datasets[i], datasets[j]] = diff_std
            df_js_std.loc[datasets[j], datasets[i]] = diff_std

    df_js.to_csv(f"./outputs/heteroAnalysis/mix/jsDists_awes{suffix}.csv",
                 header=True, index=True)
    df_js_std.to_csv(f"./outputs/heteroAnalysis/mix/std_jsDists_awes{suffix}.csv",
                     header=True, index=True)

    """ feature analysis """
    all_dict_sim = []
    for i in range(len(datasets)):
        # graphs = [list_tudatasets[i][idx] for idx in client_trainIndices[i]]
        all_dict_sim.append(_generate_distribution_nodeLabelSimilarity_mix(list_tudatasets[i]))

    df_js = pd.DataFrame(0, index=datasets, columns=datasets)
    df_js_std = pd.DataFrame(0, index=datasets, columns=datasets)
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            diff, diff_std = _get_avg_JSdist_simDistribution_byClient_mixtiny(all_dict_sim[i], all_dict_sim[j])
            print((i, j), diff, diff_std)
            df_js.loc[datasets[i], datasets[j]] = diff
            df_js.loc[datasets[j], datasets[i]] = diff
            df_js_std.loc[datasets[i], datasets[j]] = diff_std
            df_js_std.loc[datasets[j], datasets[i]] = diff_std

    df_js.to_csv(f"./outputs/heteroAnalysis/mix/jsDists_nodeLabelSim{suffix}.csv",
                 header=True, index=True)
    df_js_std.to_csv(f"./outputs/heteroAnalysis/mix/std_jsDists_nodeLabelSim{suffix}.csv",
                     header=True, index=True)

    #################################################################################################################

    # ######################## one datasets ##########################
    # dataset = 'IMDB-BINARY'
    #
    # tudataset = TUDataset(os.path.join(datapath, f"TUDataset"), dataset)
    # if not tudataset[0].__contains__('x'):
    #     tudataset = convert_to_nodeDegreeFeatures(tudataset)
    #
    #
    # """ structral similarity """
    # df_awe = pd.read_csv(f"./outputs/AWEs/mix/AWEs_{dataset}_3-7.csv", index_col=None, header=0)
    #
    # random.seed(123)
    #
    # cols = list(df_awe.columns)
    # random.shuffle(cols)
    # k = int((len(cols)+1) / 2)
    # # print('k', k)
    # cols1 = cols[:k]
    # cols2 = cols[k:]
    # # print(cols1[:10])
    # # print(cols2[:10])
    #
    # num_client = 2
    #
    # diff, diff_std = _get_avg_JSdist_awe_byClient_one(df_awe, cols1, cols2)
    # print("structure hetero", diff, diff_std)
    #
    # """ feature similarity """
    # dict_sim = _generate_distribution_nodeLabelSimilarity(tudataset)
    #
    # diff, diff_std = _get_avg_JSdist_simDistribution_byClient_one(dict_sim, cols1, cols2)
    # print("feature hetero", diff, diff_std)



if __name__ == "__main__":
    """ get the distribution of difference between end nodes of edges base on node label / node degree """
    # # # main_distribution_difference_nodeLabelsAndDegrees()
    # # # main_KStest_distDiff_nodeLabels()
    # inpath = './outputs/featureStats/'
    # dict_dist_diff_nodeLabel = json.load(open(os.path.join(inpath, "distribution_diff_nodeLabels_edgewise.txt")))
    # # # print(dict_dist_diff_nodeLabel["MUTAG"])
    # datasets = list(dict_dist_diff_nodeLabel.keys())
    # # ds_pairs = []
    # # for i in range(len(datasets)):
    # #     for j in range(i, len(datasets)):
    # #         ds_pairs.append((datasets[i], datasets[j]))
    # # with Pool(6) as p:
    # #     p.map(main_ratio_KStest_distDiff_nodeLabels, ds_pairs)
    #
    # suffix = 'nodeLabels'
    # aggregate_ratioFiles(datasets, inpath, suffix)
    #
    # inpath = './outputs/featureStats'
    # dict_dist_diff_degrs = json.load(open(os.path.join(inpath, "distribution_diff_nodeDegrees_edgewise.txt")))
    # datasets = list(dict_dist_diff_degrs.keys())
    # ds_pairs = []
    # for i in range(len(datasets)):
    #     for j in range(i, len(datasets)):
    #         ds_pairs.append((datasets[i], datasets[j]))
    # with Pool(6) as p:
    #     p.map(main_ratio_KStest_distDiff_nodeDegrees, ds_pairs)


    """ UPDATING: to use KL divergence (JS distance) """
    analyze_struct_feature_mixtiny()

    # PROTEINS
    analyze_struct_feature_proteins()

    """ Compare the clustering results between GCFL & GCFL+ """
    compare_struct_feature_mix()
    calc_similarity_clusterwise_mix()


    analyze_struct_feature_heterogeneity()