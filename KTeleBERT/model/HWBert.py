import os
import os.path as osp
import pdb
import torch
import torch.nn as nn
import numpy as np
from random import *
import json
from packaging import version
import torch.distributed as dist

from .Tool_model import AutomaticWeightedLoss
from .Numeric import AttenNumeric
from .KE_model import KE_model
# from modeling_transformer import Transformer


from .bert import BertModel, BertTokenizer, BertForMaskedLM, BertConfig
import torch.nn.functional as F

from copy import deepcopy
from src.utils import torch_accuracy
# 4.21.2


def debug(input, kk, begin=None):
    aaa = deepcopy(input[0])
    if begin is None:
        aaa.input_ids = input[0].input_ids[:kk]
        aaa.attention_mask = input[0].attention_mask[:kk]
        aaa.chinese_ref = input[0].chinese_ref[:kk]
        aaa.kpi_ref = input[0].kpi_ref[:kk]
        aaa.labels = input[0].labels[:kk]
    else:
        aaa.input_ids = input[0].input_ids[begin:kk]
        aaa.attention_mask = input[0].attention_mask[begin:kk]
        aaa.chinese_ref = input[0].chinese_ref[begin:kk]
        aaa.kpi_ref = input[0].kpi_ref[begin:kk]
        aaa.labels = input[0].labels[begin:kk]

    return aaa


class HWBert(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_awl = AutomaticWeightedLoss(args.awl_num, args)
        self.args = args
        self.config = BertConfig()
        model_name = args.model_name
        if args.model_name in ['TeleBert', 'TeleBert2', 'TeleBert3']:
            self.encoder = BertForMaskedLM.from_pretrained(osp.join(args.data_root, 'transformer', model_name))
            # MacBert来初始化 predictions layer
            if args.cls_head_init:
                tmp = BertForMaskedLM.from_pretrained(osp.join(args.data_root, 'transformer', 'MacBert'))
                self.encoder.cls.predictions = tmp.cls.predictions
        else:
            if not osp.exists(osp.join(args.data_root, 'transformer', args.model_name)):
                model_name = 'MacBert'
            self.encoder = BertForMaskedLM.from_pretrained(osp.join(args.data_root, 'transformer', model_name))
        self.numeric_model = AttenNumeric(self.args)

    # ----------------------- 主forward函数 ----------------------------------
    def forward(self, input):
        mask_loss, kpi_loss, kpi_loss_weight, kpi_loss_dict = self.mask_forward(input)
        mask_loss = mask_loss.loss
        loss_dic = {}
        if not self.args.use_kpi_loss:
            kpi_loss = None
        if kpi_loss is not None:
            loss_sum = self.loss_awl(mask_loss, 0.3 * kpi_loss)
            loss_dic['kpi_loss'] = kpi_loss.item()
        else:
            loss_sum = self.loss_awl(mask_loss)
        loss_dic['mask_loss'] = mask_loss.item()
        return {
            'loss': loss_sum,
            'loss_dic': loss_dic,
            'loss_weight': self.loss_awl.params.tolist(),
            'kpi_loss_weight': kpi_loss_weight,
            'kpi_loss_dict': kpi_loss_dict
        }

    # loss_sum, loss_dic, self.loss_awl.params.tolist(), kpi_loss_weight, kpi_loss_dict

    # ----------------------------------------------------------------
    # 测试代码，计算mask是否正确
    def mask_prediction(self, inputs, tokenizer_sz, topk=(1,)):
        token_num, token_right, word_num, word_right = None, None, None, None
        outputs, kpi_loss, kpi_loss_weight, kpi_loss_dict = self.mask_forward(inputs)
        inputs = inputs['labels'].view(-1)
        input_list = inputs.tolist()
        # 被修改的词
        change_token_index = [i for i, x in enumerate(input_list) if x != -100]
        change_token = torch.tensor(change_token_index)
        inputs_used = inputs[change_token]
        pred = outputs.logits.view(-1, tokenizer_sz)
        pred_used = pred[change_token].cpu()
        # 返回的list
        # 计算acc
        acc, token_right = torch_accuracy(pred_used, inputs_used, topk)
        # 计算混乱分数

        token_num = inputs_used.shape[0]
        # TODO: 添加word_num, word_right
        # token_right：list
        return token_num, token_right, outputs.loss.item()

    def mask_forward(self, inputs):
        kpi_ref = None
        if 'kpi_ref' in inputs:
            kpi_ref = inputs['kpi_ref']

        outputs, kpi_loss, kpi_loss_weight, kpi_loss_dict = self.encoder(
            input_ids=inputs['input_ids'].cuda(),
            attention_mask=inputs['attention_mask'].cuda(),
            # token_type_ids=inputs.token_type_ids.cuda(),
            labels=inputs['labels'].cuda(),
            kpi_ref=kpi_ref,
            kpi_model=self.numeric_model
        )
        return outputs, kpi_loss, kpi_loss_weight, kpi_loss_dict

    # TODO: 垂直注意力考虑：https://github.com/lucidrains/axial-attention

    def cls_embedding(self, inputs, tp='cls'):
        hidden_states = self.encoder(
            input_ids=inputs['input_ids'].cuda(),
            attention_mask=inputs['attention_mask'].cuda(),
            output_hidden_states=True)[0].hidden_states
        if tp == 'cls':
            return hidden_states[-1][:, 0]
        else:
            index_real = torch.tensor(inputs['input_ids'].clone().detach(), dtype=torch.bool)
            res = []
            for i in range(hidden_states[-1].shape[0]):
                if tp == 'last_avg':
                    res.append(hidden_states[-1][i][index_real[i]][:-1].mean(dim=0))
                elif tp == 'last2avg':
                    res.append((hidden_states[-1][i][index_real[i]][:-1] + hidden_states[-2][i][index_real[i]][:-1]).mean(dim=0))
                elif tp == 'last3avg':
                    res.append((hidden_states[-1][i][index_real[i]][:-1] + hidden_states[-2][i][index_real[i]][:-1] + hidden_states[-3][i][index_real[i]][:-1]).mean(dim=0))
                elif tp == 'first_last_avg':
                    res.append((hidden_states[-1][i][index_real[i]][:-1] + hidden_states[1][i][index_real[i]][:-1]).mean(dim=0))

            return torch.stack(res)
