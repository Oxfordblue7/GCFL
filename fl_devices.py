""" Adapted from https://github.com/felisat/clustered-federated-learning """

import random
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from dtaidistance import dtw


def train_gc(model, train_loader, val_loader, optimizer, local_epoch, device):
    losses_train, accs_train, losses_val, accs_val = [], [], [], []
    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0

        acc_sum = 0

        for _, batch in enumerate(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.y
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            ngraphs += batch.num_graphs
        total_loss /= ngraphs
        acc = acc_sum / ngraphs

        loss_v, acc_v = eval_gc(model, val_loader, device)

        losses_train.append(total_loss)
        accs_train.append(acc)
        losses_val.append(loss_v)
        accs_val.append(acc_v)
    return losses_train, accs_train, losses_val, accs_val

def eval_gc(model, test_loader, device):
    model.eval()

    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)
        total_loss += loss.item() * batch.num_graphs
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += batch.num_graphs

    return total_loss/ngraphs, acc_sum/ngraphs


def copy(target, source, keys):
    for name in keys:
        target[name].data = source[name].data.clone()

def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()

def reduce_add_average(targets, sources):
    for target in targets:
        for name in target:
            tmp = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()
            target[name].data += tmp

def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i, j] = torch.sum(s1 * s2) / (torch.norm(s1) * torch.norm(s2) + 1e-12)

    return angles.numpy()



class Client_GC():
    def __init__(self, model, client_id, client_name, dataLoader, optimizer, args):
        self.model = model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.args = args

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}

        self.train_stats = ([0], [0], [0], [0])
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.

    def synchronize_with_server(self, server):
        self.gconvNames = server.W.keys()
        for k in server.W:
            self.W[k].data = server.W[k].data.clone()

    def compute_weight_update(self, local_epoch):
        copy(target=self.W_old, source=self.W, keys=self.gconvNames)

        train_stats = train_gc(self.model, self.dataLoader['train'], self.dataLoader['val'], self.optimizer, local_epoch, self.args.device)

        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)

        self.train_stats = train_stats

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        dWs_conv = {key: self.dW[key] for key in self.gconvNames}
        self.convDWsNorm = torch.norm(flatten(dWs_conv)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    def reset(self):
        copy(target=self.W, source=self.W_old, keys=self.gconvNames)

    def evaluate(self):
        return eval_gc(self.model, self.dataLoader['test'], self.args.device)


class Server():
    def __init__(self, model_fn, device):
        self.model = model_fn.to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.model_cache = []

    def select_clients(self, clients, frac=1.0):
        return random.sample(clients, int(len(clients) * frac))

    def aggregate_weight_updates(self, clients):
        client_dWs = []
        for client in clients:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            client_dWs.append(dW)

        reduce_add_average(targets=self.W, sources=client_dWs)

    def compute_pairwise_similarities(self, clients):
        client_dWs = []
        for client in clients:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            client_dWs.append(dW)
        return pairwise_angles(client_dWs)

    def compute_pairwise_distances(self, seqs, standardize=False):
        """ computes DTW distances """
        if standardize:
            # standardize to only focus on the trends
            seqs = np.array(seqs)
            seqs = seqs / seqs.std(axis=1).reshape(-1, 1)
            distances = dtw.distance_matrix(seqs)
        else:
            distances = dtw.distance_matrix(seqs)
        return distances

    def cluster_clients(self, S, idc):
        clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)
        c1 = np.array([idc[x] for x in np.argwhere(clustering.labels_ == 0).flatten()])
        c2 = np.array([idc[x] for x in np.argwhere(clustering.labels_ == 1).flatten()])
        return c1, c2

    def cluster_clients_distances(self, distances, idc):
        clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(distances)
        c1 = np.array([idc[x] for x in np.argwhere(clustering.labels_ == 0).flatten()])
        c2 = np.array([idc[x] for x in np.argwhere(clustering.labels_ == 1).flatten()])
        return c1, c2

    def aggregate_clusterwise(self, client_clusters):
        for cluster in client_clusters:
            targs = []
            sours = []
            for client in cluster:
                W = {}
                dW = {}
                for k in self.W.keys():
                    W[k] = client.W[k]
                    dW[k] = client.dW[k]
                targs.append(W)
                sours.append(dW)
            reduce_add_average(targets=targs, sources=sours)


    def aggregate_weights(self, clients):
        """ Averaging the weights of all conv layers. """
        num_client = len(clients) * 1.

        for k in self.W.keys():
            tmp = clients[0].W[k].clone()
            for client in clients[1:]:
                tmp += client.W[k].clone()
            self.W[k].data = torch.div(tmp, num_client)

    def aggregate_weights_clusterwise(self, client_clusters):
        """ Averaging the weights of all conv layers based on clusters. """
        for cluster in client_clusters:
            self.aggregate_weights(cluster)

    def compute_max_update_norm(self, cluster):
        max_dW = -np.inf
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            update_norm = torch.norm(flatten(dW)).item()
            if update_norm > max_dW:
                max_dW = update_norm
        return max_dW

    def compute_mean_update_norm(self, cluster):
        cluster_dWs = []
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            cluster_dWs.append(flatten(dW))

        return torch.norm(torch.mean(torch.stack(cluster_dWs), dim=0)).item()

    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [(idcs,
                              {name: params[name].data.clone() for name in params},
                              [accuracies[i] for i in idcs])]

