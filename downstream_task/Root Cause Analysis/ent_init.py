import torch.nn as nn
import json
from fault_pkg import *
import torch
import torch.nn.functional as F
import pdb


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(hid_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        # x = F.relu(self.linear1(x))
        # x = self.linear2(x)

        return x


class EntInitAlarmFeat(nn.Module):
    def __init__(self, args):
        super(EntInitAlarmFeat, self).__init__()
        pass

    def forward(self, g):
        ent_emb = g.ndata['alarm_kpi_feat']

        return ent_emb

class EntInit(nn.Module):
    def __init__(self, args, map_dict):
        super(EntInit, self).__init__()
        self.args = args
        pretrain_emb = torch.load(args.pt_emb_path).to(args.gpu)
        self.pt_emb = nn.Parameter(pretrain_emb)

    def forward(self, g):
        ent_pt_emb = torch.mm(g.ndata['alarm_kpi_feat'], self.pt_emb) / \
                         (torch.sum(g.ndata['alarm_kpi_feat'], dim=-1).reshape(-1, 1) + 1e-6)

        return ent_pt_emb


class EntInitWithAttr(nn.Module):
    def __init__(self, args, map_dict):
        super(EntInitWithAttr, self).__init__()
        self.nftype_name2idx = map_dict['nftype_name2idx']

        self.args = args

        pretrain_emb = torch.load(args.pt_emb_path).to(args.gpu)
        self.pt_emb = nn.Parameter(pretrain_emb)

    def forward(self, g):
        ent_pt_emb = torch.mm(g.ndata['alarm_kpi_attr_feat'], self.pt_emb) / \
                        (torch.sum(g.ndata['alarm_kpi_attr_feat'], dim=-1).reshape(-1, 1) + 1e-6)

        return ent_pt_emb        

class EntInitOnlyName(nn.Module):
    def __init__(self, args, map_dict):
        super(EntInitOnlyName, self).__init__()
        self.nftype_name2idx = map_dict['nftype_name2idx']

        self.args = args

        pretrain_emb = torch.load(args.pt_emb_path).to(args.gpu)
        self.pt_emb = nn.Parameter(pretrain_emb)

    def forward(self, g):
        ent_pt_emb = torch.mm(g.ndata['alarm_kpi_feat'], self.pt_emb) / \
                        (torch.sum(g.ndata['alarm_kpi_feat'], dim=-1).reshape(-1, 1) + 1e-6)
    

        return ent_pt_emb


# class EntInit(nn.Module):
#     def __init__(self, args, map_dict):
#         super(EntInit, self).__init__()
#         self.args = args

#         self.alarm_name2idx = map_dict['alarm_name2idx']
#         self.kpi_name2idx = map_dict['kpi_name2idx']

#         alarm_list = list(json.load(open(self.args.pt_input_alarm, 'r')).values())
#         kpi_list = list(json.load(open(self.args.pt_input_kpi, 'r')).values())
#         alarm_name2ori_idx = {k: v for v, k in enumerate(alarm_list)}
#         kpi_name2ori_idx = {k: v+len(alarm_list) for v, k in enumerate(kpi_list)}

#         pretrain_emb = torch.load(args.pt_emb_path).to(args.gpu)

#         permute_alarm = torch.zeros(len(self.alarm_name2idx), dtype=torch.int64)
#         permute_kpi = torch.zeros(len(self.kpi_name2idx), dtype=torch.int64)
#         for alarm_name, idx in self.alarm_name2idx.items():
#             permute_alarm[idx] = alarm_name2ori_idx[alarm_name]
#         # for kpi_name, idx in self.kpi_name2idx.items():
#         #     permute_kpi[idx] = kpi_name2ori_idx[kpi_name]

#         self.alarm_pt_emb = nn.Parameter(pretrain_emb[permute_alarm])
#         # self.alarm_pt_emb = nn.Parameter(torch.zeros(len(alarm_name2idx), 768)).to(args.gpu)
#         # nn.init.xavier_uniform_(self.alarm_pt_emb, gain=nn.init.calculate_gain('relu'))
#         # self.kpi_pt_emb = nn.Parameter(pretrain_emb[permute_kpi])

#         # self.dense_pt_mlp = MLP(pretrain_emb.shape[1], pretrain_emb.shape[1]//2, args.ent_dim)
#         self.dense_pt_mlp = MLP(pretrain_emb.shape[1], pretrain_emb.shape[1]//2, 20)
#         self.emb_cat_mlp = MLP(args.ent_dim*2, args.ent_dim, args.ent_dim)

#         # self.nftype_emb = nn.Parameter(torch.Tensor(len(nftype_name2idx), args.ent_dim).to(args.gpu))
#         # nn.init.xavier_uniform_(self.nftype_emb, gain=nn.init.calculate_gain('relu'))

#     def forward(self, g):
#         ent_pt_emb = torch.mm(g.ndata['alarm_feat'], self.alarm_pt_emb) / \
#                          (torch.sum(g.ndata['alarm_feat'], dim=-1).reshape(-1, 1) + 1e-6)

#         return ent_pt_emb


# class EntInitWithAttr(nn.Module):
#     def __init__(self, args, map_dict):
#         super(EntInitWithAttr, self).__init__()
#         self.nftype_name2idx = map_dict['nftype_name2idx']

#         self.args = args

#         attr_list = list(json.load(open('./pretrain/fault_attr.json', 'r')).keys())
#         self.id2idx = {attr_id: idx for idx, attr_id in enumerate(attr_list)}
#         pretrain_emb = torch.load(args.pt_emb_path).to(args.gpu)

#         self.pt_emb = nn.Parameter(pretrain_emb)
#         # self.alarm_pt_emb = pretrain_emb
#         # self.pt_emb = pretrain_emb

#         # self.dense_pt_mlp = MLP(pretrain_emb.shape[1], pretrain_emb.shape[1]//2, 100)
#         self.emb_cat_mlp = MLP(pretrain_emb.shape[1]*2, pretrain_emb.shape[1], pretrain_emb.shape[1])

#         # self.nftype_emb = nn.Parameter(torch.Tensor(len(self.nftype_name2idx), 100).to(args.gpu))
#         # nn.init.xavier_uniform_(self.nftype_emb, gain=nn.init.calculate_gain('relu'))

#     def forward(self, g):
#         ent_pt_emb = torch.mm(g.ndata['alarm_kpi_attr_feat'], self.pt_emb) / \
#                          (torch.sum(g.ndata['alarm_kpi_attr_feat'], dim=-1).reshape(-1, 1) + 1e-6)

#         # ent_alarm_pt_emb = torch.mm(g.ndata['alarm_attr_feat'], self.pt_emb) / \
#         #              (torch.sum(g.ndata['alarm_attr_feat'], dim=-1).reshape(-1, 1) + 1e-6)
#         #
#         # ent_kpi_pt_emb = torch.mm(g.ndata['kpi_attr_feat'], self.pt_emb) / \
#         #              (torch.sum(g.ndata['kpi_attr_feat'], dim=-1).reshape(-1, 1) + 1e-6)

#         # ent_pt_emb = self.emb_cat_mlp(torch.cat([ent_alarm_pt_emb, ent_kpi_pt_emb], dim=-1))

#         return ent_pt_emb




