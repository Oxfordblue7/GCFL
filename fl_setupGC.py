import torch

from models import GIN, serverGIN
from fl_devices import Server, Client_GC


def setup_devices(splitedData, args):
    idx_clients = {}
    clients = []
    i = 0
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloaders, num_node_features, num_graph_labels = splitedData[ds]
        cmodel_gc = GIN(num_node_features, args.hidden, num_graph_labels, args.nlayer)
        optimizer = torch.optim.Adam(cmodel_gc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        clients.append(Client_GC(cmodel_gc, idx, ds, dataloaders, optimizer, args))
        i += 1

    smodel = serverGIN(nlayer=args.nlayer, nhid=args.hidden)
    server = Server(smodel, args.device)
    return clients, server, idx_clients