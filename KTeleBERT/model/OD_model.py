import os
import os.path as osp
import pdb
import torch
import torch.nn as nn
import numpy as np
# from transformers import BertModel, BertTokenizer, BertForMaskedLM
import json
from packaging import version
import torch.distributed as dist


class OD_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.order_num = args.order_num
        if args.od_type == 'linear_cat':
            # self.order_dense_1 = nn.Linear(args.hidden_size * self.order_num, args.hidden_size)
            # self.order_dense_2 = nn.Linear(args.hidden_size, 1)
            self.order_dense_1 = nn.Linear(args.hidden_size * self.order_num, args.hidden_size)
            if self.args.num_od_layer > 0:
                self.layer = nn.ModuleList([OD_Layer_linear(args) for _ in range(args.num_od_layer)])

        self.order_dense_2 = nn.Linear(args.hidden_size, 1)

        self.actication = nn.LeakyReLU()
        self.bn = torch.nn.BatchNorm1d(args.hidden_size)
        self.dp = nn.Dropout(p=args.hidden_dropout_prob)
        self.loss_func = nn.BCEWithLogitsLoss()
        # self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input, labels):
        # input 切成两半
        # 换方向拼接
        loss_dic = {}
        pre = self.predict(input)
        # pdb.set_trace()
        loss = self.loss_func(pre, labels.unsqueeze(1))
        loss_dic['order_loss'] = loss.item()
        return loss, loss_dic

    def encode(self, input):
        if self.args.num_od_layer > 0:
            for layer_module in self.layer:
                input = layer_module(input)
        inputs = torch.chunk(input, 2, dim=0)
        emb = torch.concat(inputs, dim=1)
        return self.actication(self.order_dense_1(self.dp(emb)))

    def predict(self, input):
        return self.order_dense_2(self.bn(self.encode(input)))

    def right_caculate(self, input, labels, threshold=0.5):
        input = input.squeeze(1).tolist()
        labels = labels.tolist()
        right = 0
        for i in range(len(input)):
            if (input[i] >= threshold and labels[i] >= 0.5) or (input[i] < threshold and labels[i] < 0.5):
                right += 1
        return right


class OD_Layer_linear(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.actication = nn.LeakyReLU()
        self.bn = torch.nn.BatchNorm1d(args.hidden_size)
        self.dropout = nn.Dropout(p=args.hidden_dropout_prob)

    def forward(self, input):
        return self.actication(self.bn(self.dense(self.dropout(input))))
