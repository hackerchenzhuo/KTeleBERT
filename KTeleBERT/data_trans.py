import os.path as osp
import numpy as np
import random
import torch
import argparse
import pdb
import json

'''
把数据合并
同时抽取一部分需要的数据出来
'''

this_dir = osp.dirname(__file__)

data_root = osp.abspath(osp.join(this_dir, '..', '..', 'data', ''))

data_path = "huawei"
data_path = osp.join(data_root, data_path)



with open(osp.join(data_path, 'product_corpus.json'), "r") as f:
    data_doc = json.load(f)

with open(osp.join(data_path, '831_alarm_serialize.json'), "r") as f:
    data_alarm = json.load(f)
# kpi_info.json
with open(osp.join(data_path, '917_kpi_serialize_50_mn.json'), "r") as f:
    data_kpi = json.load(f)


# 实体的序列化
with open(osp.join(data_path, '5GC_KB/database_entity_serialize.json'),"r") as f:
    data_entity = json.load(f)


# 百分之一的KPI数据
# data_kpi: 4389204
# data_alarm: 79501
# data_entity：96188
# data = data_alarm + data_kpi[0:int(0.015*len(data_kpi))] + data_entity

# data_doc: 425532 -> 431821
# data_kpi: 54299 -> 116587
# data_alarm: 79501
# data_entity：96188
random.shuffle(data_kpi)
random.shuffle(data_doc)
random.shuffle(data_alarm)
random.shuffle(data_entity)
data = data_alarm + data_kpi + data_entity + data_doc
random.shuffle(data)

# 241527
pdb.set_trace()
with open(osp.join(data_path, 'Seq_data_large.json'),"w") as fp:
    json.dump(data, fp, ensure_ascii=False)



# 三元组
with open(osp.join(data_path, '5GC_KB/database_triples.json'), "r") as f:
    data = json.load(f)
random.shuffle(data)


with open(osp.join(data_path, 'KG_data_base.json'),"w") as fp:
    json.dump(data, fp, ensure_ascii=False)



# ltp.pipeline(['[CLS]  [ENT] Info_Measure |   N11(SMF)发送PDU Session Establishment Reject消息数-#82 maximum data rate per UE for user-plane integrity protection is too low [ATTR] product | UNC [ATTR] labels | UNC,event [ATTR] alias |   N11(SMF)发送PDU Session Establishment Reject消息数-#82 maximum data rate per UE for user-plane integrity protection is too low; ;N11(SMF)发送PDU Session Establishment Reject消息数-#82 maximum data rate per UE for user-plane integrity protection is too low [SEP]'], tasks=["cws"])


 


# ltp.pipeline(['  is too low'], tasks=["cws"])
# protection
# [SEP]
# integrity
# for user-plane 
# for user-plane integrity protection is too low [SEP]
# maximum data rate per UE for user-plane integrity protection is too low;