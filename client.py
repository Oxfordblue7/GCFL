import torch


class Client_GC():
    def __init__(self, model, client_id, client_name, dataLoader, optimizer, args):
        self.model = model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.args = args

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}

        self.gconvNames = None

        self.train_stats = ([0], [0], [0], [0], [0])
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.

    def download_from_server(self, server):
        self.gconvNames = server.Ws[0].keys()
        for k in server.Ws[0]:
            self.W[k].data = server.Ws[0][k].data.clone()

    def download_from_server_clusterwise(self, server, idx_cluster):
        for k in server.Ws[idx_cluster]:
            self.W[k].data = server.Ws[idx_cluster][k].data.clone()

    def cache_weights(self):
        for name in self.W.keys():
            self.W_old[name].data = self.W[name].data.clone()

    def freeze_convLayers(self):
        for key, para in self.model.named_parameters():
            if key in self.gconvNames:
                para.requires_grad = False

    def unfreeze_convLayers(self):
        for key, para in self.model.named_parameters():
            if key in self.gconvNames:
                para.requires_grad = True

    def _freeze_nonConvLayers(self):
        for key, para in self.model.named_parameters():
            if key not in self.gconvNames:
                para.requires_grad = False

    def local_train(self, local_epoch):
        train_stats = train_gc(self.model, self.dataLoader['train'], self.dataLoader['val'], self.optimizer, local_epoch, self.args.device,
                               self.gconvNames, self.W)

        self.train_stats = train_stats

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()


    def evaluate(self):
        return eval_gc(self.model, self.dataLoader['test'], self.args.device)

    def local_train_prox(self, local_epoch, mu):
        train_stats = train_gc_prox(self.model, self.dataLoader['train'], self.dataLoader['val'], self.optimizer, local_epoch, self.args.device,
                               self.gconvNames, self.W, mu, self.W_old)

        self.train_stats = train_stats

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()


    def evaluate_prox(self, mu):
        return eval_gc_prox(self.model, self.dataLoader['test'], self.args.device, self.gconvNames, mu, self.W_old)



def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])


def calc_gradsNorm(gconvNames, Ws):
    grads_conv = {k: Ws[k].grad for k in gconvNames}
    convGradsNorm = torch.norm(flatten(grads_conv)).item()
    return convGradsNorm


def train_gc(model, train_loader, val_loader, optimizer, local_epoch, device, gconvNames, Ws):
    losses_train, accs_train, losses_val, accs_val = [], [], [], []
    convGradsNorm = []
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

        convGradsNorm.append(calc_gradsNorm(gconvNames, Ws))

    return losses_train, accs_train, losses_val, accs_val, convGradsNorm

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


def _prox_term(model, gconvNames, Wt):
    prox = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        # only add the prox term for sharing layers (gConv)
        if name in gconvNames:
            prox = prox + torch.norm(param - Wt[name]).pow(2)
    return prox

def train_gc_prox(model, train_loader, val_loader, optimizer, local_epoch, device, gconvNames, Ws, mu, Wt):
    losses_train, accs_train, losses_val, accs_val = [], [], [], []
    convGradsNorm = []
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
            loss = model.loss(pred, label) + mu / 2. * _prox_term(model, gconvNames, Wt)
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

        convGradsNorm.append(calc_gradsNorm(gconvNames, Ws))

    return losses_train, accs_train, losses_val, accs_val, convGradsNorm

def eval_gc_prox(model, test_loader, device, gconvNames, mu, Wt):
    model.eval()

    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label) + mu / 2. * _prox_term(model, gconvNames, Wt)
        total_loss += loss.item() * batch.num_graphs
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += batch.num_graphs

    return total_loss/ngraphs, acc_sum/ngraphs