import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class GNNLayer(nn.Module):
    def __init__(self, args, in_dim, out_dim, act=None):
        super(GNNLayer, self).__init__()
        self.args = args
        self.act = act

        self.W = nn.Linear(in_dim, out_dim)
        self.W_S = nn.Linear(in_dim, out_dim)

        self.q_linear = nn.Linear(in_dim, in_dim)
        self.k_linear = nn.Linear(in_dim, in_dim)

    def msg_func(self, edges):
        msg = self.W(edges.src['h'])
        q = self.q_linear(edges.src['h'])

        return {'msg': msg, 'q': q}

    def reduce_func(self, nodes):
        k = self.k_linear(nodes.data['h']).unsqueeze(2)
        alpha = torch.bmm(nodes.mailbox['q'], k) / np.sqrt(k.shape[1])
        alpha = F.softmax(alpha, dim=1)
        h = torch.sum(alpha * nodes.mailbox['msg'], dim=1)

        return {'h_agg': h}

    def apply_node_func(self, nodes):
        comp_h_s = nodes.data['h']

        h_new = self.W_S(comp_h_s) + nodes.data['h_agg']

        if self.act is not None:
            h_new = self.act(h_new)

        return {'h': h_new}

    def forward(self, g, ent_emb):
        with g.local_scope():
            g.ndata['h'] = ent_emb
            g.update_all(self.msg_func, self.reduce_func, self.apply_node_func)
            ent_emb = g.ndata['h']

        return ent_emb


class GNN(nn.Module):
    def __init__(self, args, in_dim, hid_dim, out_dim, nlayer=2):
        super(GNN, self).__init__()
        self.args = args
        self.layers = nn.ModuleList()
        for idx in range(nlayer):
            if idx == nlayer - 1:
                self.layers.append(GNNLayer(args, hid_dim, out_dim, act=None))
            else:
                self.layers.append(GNNLayer(args, in_dim, hid_dim, act=F.relu))

    def forward(self, g, ent_emb):
        with g.local_scope():
            for layer in self.layers:
                ent_emb = layer(g, ent_emb)
                # torch.cuda.empty_cache()

        return ent_emb
