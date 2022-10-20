from src.utils import add_special_token
import os.path as osp
import numpy as np
import random
import torch
from easydict import EasyDict as edict
import argparse
import pdb
import json
from model import BertTokenizer
from collections import Counter
from tqdm import tqdm
from time import time
from numpy import mean
import math

from transformers import BertModel


class cfg():
    def __init__(self):
        self.this_dir = osp.dirname(__file__)
        # change
        self.data_root = osp.abspath(osp.join(self.this_dir, '..', '..', 'data', ''))

    def get_args(self):
        parser = argparse.ArgumentParser()
        # seq_data_name = "Seq_data_tiny_831"
        parser.add_argument("--data_path", default="huawei", type=str, help="Experiment path")
        parser.add_argument("--update_model_name", default='MacBert', type=str, help="MacBert")
        parser.add_argument("--pretrained_model_name", default='TeleBert', type=str, help="TeleBert")
        parser.add_argument("--read_cws", default=0, type=int, help="是否需要读训练好的cws文件")
        self.cfg = parser.parse_args()

    def update_train_configs(self):
        # TODO: update some dynamic variable
        self.cfg.data_root = self.data_root
        self.cfg.data_path = osp.join(self.data_root, self.cfg.data_path)

        return self.cfg


if __name__ == '__main__':
    '''
    功能： 得到 chinese ref 文件，同时刷新训练/测试文件（仅针对序列的文本数据）
    '''
    cfg = cfg()
    cfg.get_args()
    cfgs = cfg.update_train_configs()

    # 用来被更新的，需要添加token的tokenizer
    path = osp.join(cfgs.data_root, 'transformer', cfgs.update_model_name)
    assert osp.exists(path)
    tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=True)
    tokenizer, special_token, norm_token = add_special_token(tokenizer)
    added_vocab = tokenizer.get_added_vocab()
    vocb_path = osp.join(cfgs.data_path, 'added_vocab.json')

    with open(vocb_path, 'w') as fp:
        json.dump(added_vocab, fp, ensure_ascii=False)

    vocb_description = osp.join(cfgs.data_path, 'vocab_descrip.json')
    vocb_descrip = None

    vocb_descrip = {
        "alm": "alarm",
        "ran": "ran 无线接入网",
        "mml": "MML 人机语言命令",
        "nf": "NF 独立网络服务",
        "apn": "APN 接入点名称",
        "pgw": "PGW 数据管理子系统模块",
        "lst": "LST 查询命令",
        "qos": "QoS 定制服务质量",
        "ipv": "IPV 互联网通讯协议版本",
        "ims": "IMS IP多模态子系统",
        "gtp": "GTP GPRS隧道协议",
        "pdp": "PDP 分组数据协议",
        "hss": "HSS HTTP Smooth Stream",
        "[ALM]": "alarm 告警 标记",
        "[KPI]": "kpi 关键性能指标 标记",
        "[LOC]": "location 事件发生位置 标记",
        "[EOS]": "end of the sentence 文档结尾 标记",
        "[ENT]": "实体标记",
        "[ATTR]": "属性标记",
        "[NUM]": "数值标记",
        "[REL]": "关系标记",
        "[DOC]": "文档标记"
    }

    # if osp.exists(vocb_description):
    #     with open(vocb_description, 'r') as fp:
    #         vocb_descrip = json.load(added_vocab)

    # 用来进行embedding的模型
    path = osp.join(cfgs.data_root, 'transformer', cfgs.pretrained_model_name)
    assert osp.exists(path)
    pre_tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=True)
    model = BertModel.from_pretrained(path)

    print("use the vocb_description")
    key_to_emb = {}
    for key in added_vocab.keys():
        if vocb_description is not None:
            if key in vocb_description:
                # 一部分需要描述
                key_tokens = pre_tokenizer(vocb_description[key], return_tensors='pt')
            else:
                key_tokens = pre_tokenizer(key, return_tensors='pt')
        else:
            key_tokens = pre_tokenizer(key, return_tensors='pt')

        hidden_state = model(**key_tokens, output_hidden_states=True).hidden_states
        pdb.set_trace()
        key_to_emb[key] = hidden_state[-1][:, 1:-1, :].mean(dim=1)

    emb_path = osp.join(cfgs.data_path, 'added_vocab_embedding.pt')

    torch.save(key_to_emb, emb_path)
    print(f'save to {emb_path}')
