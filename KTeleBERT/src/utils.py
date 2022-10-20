
import os
import errno
import torch
import sys
import logging
import json
from pathlib import Path
import torch.distributed as dist
import csv
import os.path as osp
from time import time
from numpy import mean
import re
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import pdb
from torch import nn



# Huggingface的实现中，自带多种warmup策略
def set_optim(opt, model_list, freeze_part=[], accumulation_step=None):
    # Bert optim
    optimizer_list, scheduler_list, named_parameters = [], [], []
    # cur_model = model.module if hasattr(model, 'module') else model
    for model in model_list:
        model_para = list(model.named_parameters())
        model_para_train, freeze_layer = [], []
        for n, p in model_para:
            if not any(nd in n for nd in freeze_part):
                model_para_train.append((n, p))
            else:
                p.requires_grad = False
                freeze_layer.append((n, p))
        named_parameters.extend(model_para_train) 

    # for name, param in model_list[0].named_parameters():
    #     if not param.requires_grad:
    #         print(name, param.size())
            
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # numeric_model 也包括到这个部分中
    ke_part = ['ke_model', 'loss_awl', 'numeric_model', 'order']
    if opt.LLRD:
        # 按层次衰减的学习率
        all_name_orig = [n for n, p in named_parameters if not any(nd in n for nd in ke_part)]

        opt_parameters, all_name = LLRD(opt, named_parameters, no_decay, ke_part)
        remain = list(set(all_name_orig) - set(all_name))
        remain_parameters = [
                {'params': [p for n, p in named_parameters if not any(nd in n for nd in no_decay) and n in remain], "lr": opt.lr, 'weight_decay': opt.weight_decay},
                {'params': [p for n, p in named_parameters if any(nd in n for nd in no_decay) and n in remain], "lr": opt.lr, 'weight_decay': 0.0}
            ]
        opt_parameters.extend(remain_parameters)
    else:
        opt_parameters = [
                {'params': [p for n, p in named_parameters if not any(nd in n for nd in no_decay) and not any(nd in n for nd in ke_part)], "lr": opt.lr, 'weight_decay': opt.weight_decay},
                {'params': [p for n, p in named_parameters if any(nd in n for nd in no_decay) and not any(nd in n for nd in ke_part)], "lr": opt.lr, 'weight_decay': 0.0}
            ]

    ke_parameters = [
        {'params': [p for n, p in named_parameters if not any(nd in n for nd in no_decay) and any(nd in n for nd in ke_part)], "lr": opt.ke_lr, 'weight_decay': opt.weight_decay},
        {'params': [p for n, p in named_parameters if any(nd in n for nd in no_decay) and any(nd in n for nd in ke_part)], "lr": opt.ke_lr, 'weight_decay': 0.0}
    ]
    opt_parameters.extend(ke_parameters)
    optimizer = AdamW(opt_parameters, lr=opt.lr, eps=opt.adam_epsilon)
    if accumulation_step is None:
        accumulation_step = opt.accumulation_steps
    if opt.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(opt.warmup_steps/accumulation_step), num_training_steps=int(opt.total_steps/accumulation_step))
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(opt.warmup_steps/accumulation_step), num_training_steps=int(opt.total_steps/accumulation_step))
    
    # ---- 判定所有参数是否被全部优化 ----
    all_para_num = 0
    for paras in opt_parameters:
        all_para_num += len(paras['params'])
    # pdb.set_trace()
    assert len(named_parameters) == all_para_num
    return optimizer, scheduler

# LLRD 学习率逐层衰减但

def LLRD(opt, named_parameters, no_decay, ke_part =[]):
    opt_parameters = []
    all_name = []
    head_lr = opt.lr * 1.05
    init_lr = opt.lr
    lr = init_lr

    # === Pooler and regressor ======================================================  
    params_0 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n or "predictions" in n) 
                and any(nd in n for nd in no_decay) and not any(nd in n for nd in ke_part)]
    params_1 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n or "predictions" in n)
                and not any(nd in n for nd in no_decay) and not any(nd in n for nd in ke_part)]

    name_0 = [n for n,p in named_parameters if ("pooler" in n or "regressor" in n or "predictions" in n) 
                and any(nd in n for nd in no_decay) and not any(nd in n for nd in ke_part)]
    name_1 = [n for n,p in named_parameters if ("pooler" in n or "regressor" in n or "predictions" in n)
                and not any(nd in n for nd in no_decay) and not any(nd in n for nd in ke_part)]

    all_name.extend(name_0)
    all_name.extend(name_1)
    
    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}    
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": head_lr, "weight_decay": 0.01}    
    opt_parameters.append(head_params)
    
    # === 12 Hidden layers ==========================================================
    for layer in range(11,-1,-1):        
        params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay) and not any(nd in n for nd in ke_part)]
        params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay) and not any(nd in n for nd in ke_part)]

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)   

        layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
        opt_parameters.append(layer_params) 

        name_0 = [n for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay) and not any(nd in n for nd in ke_part)]
        name_1 = [n for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay) and not any(nd in n for nd in ke_part)]
        all_name.extend(name_0)
        all_name.extend(name_1)      

        lr *= 0.95 
    # === Embeddings layer ==========================================================
    
    params_0 = [p for n,p in named_parameters if ("embeddings" in n )  
                and any(nd in n for nd in no_decay) and not any(nd in n for nd in ke_part)]
    params_1 = [p for n,p in named_parameters if ("embeddings" in n ) 
                and not any(nd in n for nd in no_decay) and not any(nd in n for nd in ke_part)]
    
    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01} 
    opt_parameters.append(embed_params)   

    name_0 = [n for n,p in named_parameters if ("embeddings" in n ) 
                and any(nd in n for nd in no_decay) and not any(nd in n for nd in ke_part)]
    name_1 = [n for n,p in named_parameters if ("embeddings" in n )
                and not any(nd in n for nd in no_decay) and not any(nd in n for nd in ke_part)]
    all_name.extend(name_0)
    all_name.extend(name_1) 
    return opt_parameters, all_name

class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        return 1.0


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, min_ratio, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        # self.fixed_lr = fixed_lr
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio) * step / float(max(1, self.warmup_steps)) + self.min_ratio

        # if self.fixed_lr:
        #     return 1.0

        return max(0.0,
                   1.0 + (self.min_ratio - 1) * (step - self.warmup_steps) / float(max(1.0, self.scheduler_steps - self.warmup_steps)),
                   )


class Loss_log():
    def __init__(self):
        self.loss = []
        self.acc = [0.]
        self.flag = 0
        self.token_right_num = []
        self.token_all_num = []
        self.word_right_num = []
        self.word_all_num = []
        # 默认不使用top_k acc
        self.use_top_k_acc = 0

    def acc_init(self, topn=[1]):
        self.loss = []
        self.token_right_num = []
        self.token_all_num = []
        self.topn = topn
        self.use_top_k_acc = 1
        self.top_k_word_right = {}
        for n in topn:
            self.top_k_word_right[n] = []

    def time_init(self):
        self.start = time()
        self.last = self.start
        self.time_used_epoch = []

    def time_cpt(self, step, total_step):
        # 时间统计
        time_used_last_epoch = time()  - self.last
        self.time_used_epoch.append(time_used_last_epoch)
        time_used = time() - self.start
        self.last = time()
        h, m, s = time_trans(time_used)
        time_remain = int(total_step - step) * mean(self.time_used_epoch)
        h_r, m_r, s_r = time_trans(time_remain)

        return h, m, s, h_r, m_r, s_r

    def get_token_acc(self):
        # 返回list
        if len(self.token_all_num) == 0:
            return 0.
        elif self.use_top_k_acc == 1:
            res = []
            for n in self.topn:
                res.append(round((sum(self.top_k_word_right[n]) / sum(self.token_all_num)) * 100 , 3))
            return res
        else:
            return [sum(self.token_right_num)/sum(self.token_all_num)]
        

    def update_token(self, token_num, token_right):
        # 输入是list文件
        self.token_all_num.append(token_num)
        if isinstance(token_right, list):
            for i, n in enumerate(self.topn):
                self.top_k_word_right[n].append(token_right[i])
        self.token_right_num.append(token_right)

    def update(self, case):
        self.loss.append(case)

    def update_acc(self, case):
        self.acc.append(case)

    def get_loss(self):
        if len(self.loss) == 0:
            return 500.
        return mean(self.loss)

    def get_acc(self):
        return self.acc[-1]

    def get_min_loss(self):
        return min(self.loss)

    def early_stop(self):
        # min_loss = min(self.loss)
        if self.loss[-1] > min(self.loss):
            self.flag += 1
        else:
            self.flag = 0

        if self.flag > 1000:
            return True
        else:
            return False


def add_special_token(tokenizer, model=None, rank=0, cache_path = None):
    # model: bert layer
    # 每次更新这个，所有模型需要重新训练，get_chinese_ref.py需要重新运行
    # 主函数调用该函数的位置需要在载入模型之前
    # ---------------------------------------
    # 不会被mask的 token， 不参与 任何时候的MASK
    special_token = ['[SEP]', '[MASK]', '[ALM]', '[KPI]', '[CLS]', '[LOC]', '[EOS]', '[ENT]', '[ATTR]', '[NUM]', '[REL]', '|', '[DOC]'] 

    # ---------------------------------------
    # 会被mask的但是---#不加入#---tokenizer的内容
    # 出现次数多（>10000）但是长度较长(>=4符)
    # 或者是一些难以理解的名词
    # WWM 的主体
    # TODO: 专家检查
        # To Add： 'SGSN', '3GPP', 'Bearer', 'sbim', 'FusionSphere',  'IMSI', 'GGSN', 'RETCODE', 'PCRF', 'PDP', 'GTP', 'OCS', 'HLR', 'FFFF', 'VLR', 'DNN', 'PID', 'CSCF', 'PDN', 'SCTP', 'SPGW', 'TAU', 'PCEF', 'NSA', 'ACL', 'BGP', 'USCDB', 'VoLTE', 'RNC', 'GPRS', 'DRA', 'MOC'
        # 拆分：配置原则，本端规划
    norm_token = ['网元实例', '事件类型', '告警级别', '告警名称', '告警源', '通讯系统', '默认值', '链路故障', '取值范围', '可选必选说明', '数据来源', '用户平面', '配置', '原则',  '该参数', '失败次数', '可选参数', 'S1模式', '必选参数',  'IP地址', '响应消息', '成功次数', '测量指标', '用于', '统计周期', '该命令', '上下文', '请求次数', '本端',  'pod', 'amf', 'smf', 'nrf', 'ausf', 'upcf', 'upf', 'udm', 'PDU', 'alias', 'PLMN', 'MML', 'Info_Measure', 'icase', 'Diameter', 'MSISDN', 'RAT', 'RMV', 'PFCP', 'NSSAI', 'CCR', 'HDBNJjs', 'HNGZgd', 'SGSN', '3GPP', 'Bearer', 'sbim', 'FusionSphere',  'IMSI', 'GGSN', 'RETCODE', 'PCRF', 'PDP', 'GTP', 'OCS', 'HLR', 'FFFF', 'VLR', 'DNN', 'PID', 'CSCF', 'PDN', 'SCTP', 'SPGW', 'TAU', 'PCEF', 'NSA', 'ACL', 'BGP', 'USCDB', 'VoLTE', 'RNC', 'GPRS', 'DRA', 'MOC', '告警', '网元', '对端', '信令', '话单', '操作', '风险', '等级', '下发', '流控', '运营商', '寻呼', '漫游', '切片', '报文', '号段', '承载', '批量', '导致', '原因是', '影响', '造成', '引起', '随之', '情况下', '根因', 'trigger']
    # ---------------------------------------
    # , '', '', '', '', '', '', '', '', '', '', ''
    # 会被mask的但是---#加入#---tokenizer的内容
    # 长度小于等于3，缩写/专有名词 大于10000次
    # 严谨性要求大于norm_token
    # 出现次数多时有足够的影响力可以进行分离
    norm_token_tobe_added = ['pod', 'amf', 'smf', 'nrf', 'ausf', 'upcf', 'upf', 'udm', 'ALM', '告警', '网元', '对端', '信令', '话单', 'RAN', 'MML', 'PGW', 'MME', 'SGW', 'NF', 'APN', 'LST', 'GW', 'QoS', 'IPv', 'PDU', 'IMS', 'EPS', 'GTP', 'PDP', 'LTE', 'HSS']

    token_tobe_added = []
    # all_token = special_token + norm_token_tobe_added
    all_token = norm_token_tobe_added
    for i in all_token:
        if i not in  tokenizer.vocab.keys() and i.lower() not in  tokenizer.vocab.keys():
            token_tobe_added.append(i)

    # tokenizer.add_tokens(special_token, special_tokens=False)
    # tokenizer.add_tokens(norm_token, special_tokens=False)
    tokenizer.add_tokens(token_tobe_added, special_tokens=False)
    special_tokens_dict = {"additional_special_tokens": special_token}
    special_token_ = tokenizer.add_special_tokens(special_tokens_dict)
    if rank == 0:
        print("Added tokens:")
        print(tokenizer.get_added_vocab())
    
    # pdb.set_trace()

    if model is not None:
        # TODO: 用预训练好的TeleBert进行这部分embedding（所有添加的embedding）的初始化
        if rank == 0:
            print(f"--------------------------------")
            print(f"--------    orig word embedding shape: {model.get_input_embeddings().weight.shape}")
        sz = model.resize_token_embeddings(len(tokenizer)) 
        if cache_path is not None:
            # model.cpu()
            token_2_emb = torch.load(cache_path)
            # 在这里加入embedding 初始化之后需要tie一下 
            token_dic = tokenizer.get_added_vocab()
            id_2_token = {v:k   for k,v in token_dic.items()}
            with torch.no_grad():
                for key in id_2_token.keys():
                    model.bert.embeddings.word_embeddings.weight[key,:] = nn.Parameter(token_2_emb[id_2_token[key]][0]).cuda()
                    # model.get_input_embeddings().weight[key,:] = nn.Parameter(token_2_emb[id_2_token[key]][0]).cuda()
                # model.embedding
            model.bert.tie_weights()
        if rank == 0:
            print(f"--------    resize_token_embeddings into {sz} done!")
            print(f"--------------------------------")
        # 这里替换embedding
        
    norm_token = list(set(norm_token).union(set(norm_token_tobe_added)))
    return tokenizer, special_token, norm_token


def time_trans(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), int(s)

def torch_accuracy(output, target, topk=(1,)):
    '''
    param output, target: should be torch Variable
    '''
    # assert isinstance(output, torch.cuda.Tensor), 'expecting Torch Tensor'
    # assert isinstance(target, torch.Tensor), 'expecting Torch Tensor'
    # print(type(output))

    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, True, True) # 返回(values,indices）其中indices就是预测类别的值，0为第一类
    pred = pred.t() # torch.t()转置，既可得到每一行为batch最好的一个预测序列

    is_correct = pred.eq(target.view(1, -1).expand_as(pred))

    ans = []
    ans_num = []
    for i in topk:
        # is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim=True)
        is_correct_i = is_correct[:i].contiguous().view(-1).float().sum(0, keepdim=True)
        ans_num.append(int(is_correct_i.item()))
        ans.append(is_correct_i.mul_(100.0 / batch_size))

    return ans, ans_num

    