import torch
import numpy as np
import random
from sklearn.cluster import AgglomerativeClustering
from dtaidistance import dtw

class Server():
    def __init__(self, model, device):
        self.model = model.to(device)
        self.Ws = [{key: value for key, value in self.model.named_parameters()}]

    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_weights(self, selected_clients):
        for k in self.Ws[0].keys():
            self.Ws[0][k].data = torch.mean(torch.stack([client.W[k].data for client in selected_clients]), dim=0).clone()

    def add_new_model(self, num):
        for i in range(num):
            self.Ws.append({key: value for key, value in self.model.named_parameters()})

    def decrease_model(self, num):
        self.Ws = self.Ws[:-num]

    def aggregate_weights_clusterwise(self, cluster, idx_cluster):
        """ aggregate the weights of clients in a cluster """
        for k in self.Ws[idx_cluster].keys():
            self.Ws[idx_cluster][k].data = torch.mean(torch.stack([client.W[k].data for client in cluster]), dim=0).clone()

    def compute_distances_pairwise(self, seqs, standardize=False):
        """ compare pairwise distances among clients """
        if standardize:
            # standardize to only focus on the trends
            seqs = np.array(seqs)
            seqs = seqs / seqs.std(axis=1).reshape(-1, 1)
            distances = dtw.distance_matrix(seqs)
        else:
            distances = dtw.distance_matrix_fast(seqs)

        return distances

    def bicluster_by_distances(self, distances, cids):
        """ bi-cluster clients based on distance matrix """
        clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(distances)
        c1 = np.array([cids[x] for x in np.argwhere(clustering.labels_ == 0).flatten()])
        c2 = np.array([cids[x] for x in np.argwhere(clustering.labels_ == 1).flatten()])
        return [c1, c2]
