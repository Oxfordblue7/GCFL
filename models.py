import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GINConv, global_add_pool

class serverGIN(torch.nn.Module):
    def __init__(self, nlayer, nhid):
        super(serverGIN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
                                       torch.nn.Linear(nhid, nhid))
        self.convs.append(GINConv(self.nn1))
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
                                           torch.nn.Linear(nhid, nhid))
            self.convs.append(GINConv(self.nnk))


class GIN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer):
        super(GIN, self).__init__()
        self.num_layers = nlayer

        self.pre_mp = torch.nn.Sequential(
            torch.nn.Linear(nfeat, nhid))

        self.convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.convs.append(GINConv(self.nn1))
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.convs.append(GINConv(self.nnk))

        self.post_mp = torch.nn.Sequential(
            torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nclass))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.pre_mp(x)
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        x = self.post_mp(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
