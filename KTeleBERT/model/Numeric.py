import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np
import pdb
import math
from .Tool_model import AutomaticWeightedLoss
import os.path as osp
import json


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda())**2).sum()


class AttenNumeric(nn.Module):
    def __init__(self, config):
        super(AttenNumeric, self).__init__()
        # -----------  加载kpi2id --------------------
        kpi_file_path = osp.join(config.data_path, 'kpi2id.json')

        with open(kpi_file_path, 'r') as f:
            # pdb.set_trace()
            kpi2id = json.load(f)
        config.num_kpi = 303
        # config.num_kpi = len(kpi2id)
        # -------------------------------
        # self.num_attention_heads = config.num_attention_heads
        # self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 96
        self.config = config
        self.fc = nn.Linear(1, config.hidden_size)
        # self.actication = nn.ReLU()
        self.actication = nn.LeakyReLU()
        # self.embedding = nn.Linear(config.hidden_size, self.attention_head_size)
        if config.contrastive_loss:
            self.loss_awl = AutomaticWeightedLoss(3, config)
        else:
            self.loss_awl = AutomaticWeightedLoss(2, config)
        self.encoder = AttNumEncoder(config)
        self.decoder = AttNumDecoder(config)
        self.classifier = NumClassifier(config)
        self.ce_loss = nn.CrossEntropyLoss()

    def contrastive_loss(self, hidden, kpi):
        # in batch negative
        bs_tmp = hidden.shape[0]
        eye = torch.eye(bs_tmp).cuda()
        hidden = F.normalize(hidden, dim=1)
        # [12,12]
        # 减去对角矩阵目的是防止对自身的相似程度影响了判断
        hidden_sim = (torch.matmul(hidden, hidden.T) - eye) / 0.07
        kpi = kpi.expand(-1, bs_tmp)
        kpi_sim = torch.abs(kpi - kpi.T) + eye
        # kpi_sim = eye[torch.min(kpi_sim, 1)[1]].cuda()
        kpi_sim = torch.min(kpi_sim, 1)[1]
        # pdb.set_trace()
        sc_loss = self.ce_loss(hidden_sim, kpi_sim)
        # cls_loss = self.ce_loss(hidden_sim, kpi_id)
        # hidden_sim_ = hidden_sim[:11,:11].cuda()
        # kpi_sim_ = kpi_sim[:11].cuda()
        return sc_loss

    def _encode(self, kpi, query):
        kpi_emb = self.actication(self.fc(kpi))
        # name_emb = self.embedding(query)
        hidden, en_loss, scalar_list = self.encoder(kpi_emb, query)
        # pdb.set_trace()
        # 两个及以下的对比学习没有意义
        if self.config.contrastive_loss and hidden.shape[0] > 2:
            con_loss = self.contrastive_loss(hidden.squeeze(1), kpi.squeeze(1))
            # pdb.set_trace()
        else:
            con_loss = None
        hidden = self.actication(hidden)
        # pdb.set_trace()
        assert query.shape[0] > 0
        return hidden, en_loss, scalar_list, con_loss

    def forward(self, kpi, query, kpi_id):
        hidden, en_loss, scalar_list, con_loss = self._encode(kpi, query)
        dec_kpi_score, de_loss = self.decoder(kpi, hidden)
        cls_kpi, cls_loss = self.classifier(hidden, kpi_id)
        # loss_sum = self.loss_awl(de_loss, cls_loss)
        # return hidden, dec_kpi, cls_kpi
        # if self.config.contrastive_loss:
        if con_loss is not None:
            # 0.001 * con_loss
            loss_sum = self.loss_awl(de_loss, cls_loss, 0.1 * con_loss)
            loss_all = loss_sum + en_loss
            loss_dic = {'cls_loss': cls_loss.item(), 'reg_loss': de_loss.item(), 'orth_loss': en_loss.item(), 'con_loss': con_loss.item()}
            # pdb.set_trace()
        else:
            loss_sum = self.loss_awl(de_loss, cls_loss)
            loss_all = loss_sum + en_loss
            loss_dic = {'cls_loss': cls_loss.item(), 'reg_loss': de_loss.item(), 'orth_loss': en_loss.item()}
        # pdb.set_trace()
        # print(self.loss_awl.params)

        return dec_kpi_score, cls_kpi, hidden, loss_all, self.loss_awl.params.tolist(), loss_dic, scalar_list


class AttNumEncoder(nn.Module):
    def __init__(self, config):
        super(AttNumEncoder, self).__init__()
        self.num_l_layers = config.l_layers
        self.layer = nn.ModuleList([AttNumLayer(config) for _ in range(self.num_l_layers)])

    def forward(self, kpi_emb, name_emb):
        loss = 0.
        scalar_list = []
        for layer_module in self.layer:
            kpi_emb, orth_loss, scalar = layer_module(kpi_emb, name_emb)
            loss += orth_loss
            scalar_list.append(scalar)
        return kpi_emb, loss, scalar_list


class AttNumDecoder(nn.Module):
    def __init__(self, config):
        super(AttNumDecoder, self).__init__()
        self.dense_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_2 = nn.Linear(config.hidden_size, 1)
        self.actication = nn.LeakyReLU()
        self.loss_func = nn.MSELoss(reduction='mean')

    def forward(self, kpi_label, hidden):
        # 修复异常值
        pre = self.actication(self.dense_2(self.actication(self.dense_1(hidden))))
        loss = self.loss_func(pre, kpi_label)
        # pdb.set_trace()
        return pre, loss


class NumClassifier(nn.Module):
    def __init__(self, config):
        super(NumClassifier, self).__init__()
        self.dense_1 = nn.Linear(config.hidden_size, int(config.hidden_size / 3))
        self.dense_2 = nn.Linear(int(config.hidden_size / 3), config.num_kpi)
        self.loss_func = nn.CrossEntropyLoss()
        # self.actication = nn.ReLU()
        self.actication = nn.LeakyReLU()

    def forward(self, hidden, kpi_id):
        hidden = self.actication(self.dense_1(hidden))
        pre = self.actication(self.dense_2(hidden)).squeeze(1)
        loss = self.loss_func(pre, kpi_id)
        return pre, loss


class AttNumLayer(nn.Module):
    def __init__(self, config):
        super(AttNumLayer, self).__init__()
        self.config = config
        # 768 / 8 = 8
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 96
        # self.head_size = config.hidden_size

        # scaler
        self.scalar = nn.Parameter(.3 * torch.ones(1, requires_grad=True))
        self.key = nn.Parameter(torch.empty(self.num_attention_heads, self.attention_head_size))

        # name embedding
        self.embedding = nn.Linear(config.hidden_size, self.attention_head_size)
        # num_attention_heads�� value���� ת������k��
        self.value = nn.Linear(config.hidden_size, config.hidden_size * self.num_attention_heads)

        # add & norm
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        # 0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # for m in self.modules().modules():
        #     pdb.set_trace()

        nn.init.kaiming_normal_(self.key, mode='fan_out', nonlinearity='leaky_relu')
        # nn.init.orthogonal_(self.key)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.config.hidden_size,
        )
        x = x.view(*new_x_shape)
        return x
        # return x.permute(0, 2, 1, 3)

    def forward(self, kpi_emb, name_emb):
        # [64, 1, 96]
        name_emb = self.embedding(name_emb)

        mixed_value_layer = self.value(kpi_emb)

        # [64, 1, 8, 768]
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # name_emb �� key��
        # key: [8, 96] self.key.transpose(-1, -2): [96, 8]
        # name_emb: [64, 1, 96]
        attention_scores = torch.matmul(name_emb, self.key.transpose(-1, -2))
        # [64, 1, 8]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        attention_probs = attention_probs.unsqueeze(1)
        # ��Ȩ��value��
        # [64, 1, 1, 8] * [64, 1, 8, 768] = [64, 1, 8, 96]
        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.config.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # add & norm
        output_emb = self.dense(context_layer)
        output_emb = self.dropout(output_emb)
        output_emb = self.LayerNorm(output_emb + self.scalar * kpi_emb)
        # output_emb = self.LayerNorm(self.LayerNorm(output_emb) + self.scalar * kpi_emb)

        # pdb.set_trace()
        wei = self.value.weight.chunk(8, dim=0)
        orth_loss_value = sum([ortho_penalty(k) for k in wei])
        # 0.01 * ortho_penalty(self.key) + ortho_penalty(self.value.weight)
        orth_loss = 0.0001 * orth_loss_value + 0.0001 * ortho_penalty(self.dense.weight) + 0.01 * ((self.scalar[0])**2).sum()
        # pdb.set_trace()
        # if orth_loss.item() < 50:
        #     pdb.set_trace()
        return output_emb, orth_loss, self.scalar.tolist()[0]
