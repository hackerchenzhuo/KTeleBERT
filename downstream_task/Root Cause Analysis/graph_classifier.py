from fault_pkg import *
import torch.nn as nn
import torch.nn.functional as F
import torch


# class GraphClassifier(nn.Module):
#     def __init__(self, args):
#         super(GraphClassifier, self).__init__()
#         self.args = args
#         global label_name2idx
#
#         self.linear1 = nn.Linear(self.args.ent_dim, self.args.ent_dim // 2)
#         self.linear2 = nn.Linear(self.args.ent_dim // 2, len(label_name2idx))
#
#     def forward(self, graph_repr):
#         x = F.relu(self.linear1(graph_repr))
#         x = self.linear2(x)
#
#         return x


class NodeClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(NodeClassifier, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.sigmoid(self.bn2(self.linear2(x)))

        return x