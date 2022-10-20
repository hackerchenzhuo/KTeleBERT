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
from ltp import LTP
from tqdm import tqdm
from src.utils import add_special_token
from functools import reduce
from time import time
from numpy import mean
import math

from src.utils import Loss_log, time_trans
from collections import defaultdict


class cfg():
    def __init__(self):
        self.this_dir = osp.dirname(__file__)
        # change
        self.data_root = osp.abspath(osp.join(self.this_dir, '..', '..', 'data', ''))

    def get_args(self):
        parser = argparse.ArgumentParser()
        # seq_data_name = "Seq_data_tiny_831"
        parser.add_argument("--data_path", default="huawei", type=str, help="Experiment path")
        # TODO: freq 可以考虑 150
        parser.add_argument("--freq", default=50, type=int, help="出现多少次的词认为是重要的")
        parser.add_argument("--batch_size", default=100, type=int, help="分词的batch size")
        parser.add_argument("--seq_data_name", default='Seq_data_large', type=str, help="seq_data 名字")
        parser.add_argument("--deal_numeric", default=0, type=int, help="是否处理数值数据")

        parser.add_argument("--read_cws", default=0, type=int, help="是否需要读训练好的cws文件")
        self.cfg = parser.parse_args()

    def update_train_configs(self):
        # TODO: update some dynamic variable
        self.cfg.data_root = self.data_root
        self.cfg.data_path = osp.join(self.data_root, self.cfg.data_path)

        return self.cfg


def refresh_data(ref, freq, special_token):
    '''
    功能：在自定义的special token基础上基于最小出现频率得到更多新词分词系统的参考，作为wwm基础
    输入：
        freq: 在（37万）语义词典中的最小出现频率（空格为分词）
        special_token: 前面手工定义的特殊token（可能存在交集）
    输出：
        add_words：在定义的最小出现频率基础上筛选出来的新词
    '''
    # 经常出现的sub token
    seq_sub_data = [line.split() for line in ref]
    all_data = []
    for data in seq_sub_data:
        all_data.extend(data)
    sub_word_times = dict(Counter(all_data))
    asub_word_time_order = sorted(sub_word_times.items(), key=lambda x: x[1], reverse=True)
    # ('LST', 1218), ('RMV', 851), ('DSP', 821), ('ADD', 820), ('MOD', 590), ('SET', 406), ('AWS', 122)
    # ADD、ACT、ALM-XXX、DEL、DSP、LST
    add_words = []

    for i in asub_word_time_order:
        # 把出现频率很高的词加进来
        if i[1] >= freq and len(i[0]) > 1 and len(i[0]) < 20 and not str.isdigit(i[0]):
            add_words.append(i[0])
    add_words.extend(special_token)
    # 卡100阈值时是935个特殊token
    print(f"[{len(add_words)}] special words will be added with frequency [{freq}]!")
    return add_words


def cws(seq_data, add_words, batch_size):
    '''
    功能：所有序列数据的输入转换成分词之后的结果 
    输入：
        seq_data：所有序列数据输入 e.g.['KPI异常下降', 'KPI异常上升']  
        add_words：添加的special words
        batch_size：每次分多少句
    输出：
        all_segment：所有序列数据的输出 e.g. [['KPI', '异常', '下降']， ['KPI', '异常', '上升']] 
        data_size：输入/输出的序列数量（e.g. 2）
    '''
    # seq_data = seq_data.cuda()
    print(f"loading...")
    ltp = LTP("LTP/base2")  # 默认加载 base2 模型
    # ltp = LTP()
    print(f"begin adding words ...")
    # ltp.add_words(words=add_words, max_window=5) #4.1.5
    ltp.add_words(words=add_words)  # 4.2.8
    ltp.to("cuda")
    # for word in add_words:
    #     ltp.add_word(word)
    print(f"{len(add_words)} special words are added!")

    #
    # for data in seq_data:
    #     output = ltp.pipeline([data], tasks=["cws"])
    data_size = len(seq_data)
    seq_data_cws = []
    size = int(data_size / batch_size) + 1
    b = 0
    e = b + batch_size
    # pdb.set_trace()

    log = Loss_log()

    with tqdm(total=size) as _tqdm:
        # pdb.set_trace()
        # log.time_init()
        # pdb.set_trace()
        error_data = []
        for i in range(size):

            output = []
            try:
                _output = ltp.pipeline(seq_data[b:e], tasks=["cws"])
                for data in _output.cws:
                    try:
                        data_out = ltp.pipeline(data, tasks=["cws"])
                        # data_out_ = reduce(lambda x, y: x.extend(y) or x, data_out.cws)
                        data_out_ = []
                        for i in data_out.cws:
                            data_out_.extend([k.strip() for k in i])
                        output.append(data_out_)
                    except:
                        print(f"二阶段分词出错！范围是：[{b}]-[{e}]")
                        error_data.append(data)

            # pdb.set_trace()
            except:
                print(f"第一阶段分词出错！范围是：[{b}]-[{e}]")
                error_data.append(f"第一阶段分词出错！范围是：[{b}]-[{e}]")
                # continue
            seq_data_cws.extend(output)
            b = e
            e += batch_size

            # 时间统计
            if e >= data_size:
                if b >= data_size:
                    break
                e = data_size
            _tqdm.set_description(f'from {b} to {e}:')
            _tqdm.update(1)

    print(f"过滤了{data_size - len(seq_data_cws)}个句子")

    return seq_data_cws, data_size, error_data


def ltp_debug(ltp, op):
    output = []
    for data in op:
        data_out = ltp.pipeline(data, tasks=["cws"])
        # data_out_ = reduce(lambda x, y: x.extend(y) or x, data_out.cws)
        data_out_ = []
        for i in data_out.cws:
            # 保留空格的话需要手动去除空格
            data_out_.append(i[0].strip())
            # 之前没有空格
            # data_out_.extend(i)
        output.append(data_out_)
    return output


def deal_sub_words(subwords, special_token):
    '''
    功能：把每个word的整体内，非首字符的部分加上 '##' 前缀， special_token 不应该被mask
    '''
    for i in range(len(subwords)):
        if i == 0:
            continue
        if subwords[i] in special_token:
            continue
        if subwords[i].startswith("##"):
            continue

        subwords[i] = "##" + subwords[i]
    return subwords


def generate_chinese_ref(seq_data_cws, special_token, deal_numeric, kpi_dic):
    '''
    输入： 
        seq_data_cws：所有序列数据的输出 e.g. [['KPI', '异常', '下降']， ['KPI', '异常', '上升']] 
        special_token：不应该被mask ['[SEP]', '[MASK]', '[ALM]', '[KPI]', '[CLS]', '[LOC]', '[EOS]', '[ENT]', '[ATTR]', '[NUM]', '|']
        data_size：数据量 e.g. 2
    输出：
        ww_return （whole word return）：打标之后的chinese ref e.g. [['KPI', '异','##常', '下', '##降']， ['KPI', '异', '##常', '上', '##升']] 
    '''
    # 定义全局set和逆字典统计哪些KPI最后没有被涉及
    data_size = len(seq_data_cws)
    kpi_static_set = set()
    rev_kpi_dic = dict(zip(kpi_dic.values(), kpi_dic.keys()))
    max_len = 0
    sten_that_over_maxl = []
    with tqdm(total=data_size) as _tqdm:
        ww_return = []
        ww_list = []
        kpi_info = []
        not_in_KPI = defaultdict(int)
        for i in range(data_size):
            _tqdm.set_description(f'checking...[{i}/{data_size}] max len: [{max_len}]')
            orig = tokenizer.tokenize(" ".join(seq_data_cws[i]))

            if deal_numeric:
                # 得到元组信息，前两位是KPI下标范围
                _kpi_info, kpi_type_list = extract_kpi(orig, kpi_dic, not_in_KPI)
                kpi_info.append(_kpi_info)
                kpi_static_set.update(kpi_type_list)

            sub_total = []
            ww_seq_tmp = []
            ww_tmp = []
            for sub_data in seq_data_cws[i]:
                sub = tokenizer.tokenize(sub_data)
                sub_total.extend(sub)
                # 在whole word 里面添加#号
                # 输入:  ['异', '常']
                ref_token = deal_sub_words(sub, special_token)
                # 输出:  ['异', '##常']
                ww_seq_tmp.extend(ref_token)
                ww_tmp.append(ref_token)

            if sub_total != orig:
                print("error in match... ")
                if len(orig) > 512:
                    print("the lenth is over the max lenth")
                pdb.set_trace()

            # 变成[[...],[...],[...], ...]
            # ww_return.append(ww_tmp)
            sz_ww_seq = len(ww_seq_tmp)
            # 求最大长度
            max_len = sz_ww_seq if sz_ww_seq > max_len else max_len
            if sz_ww_seq > 500:
                sten_that_over_maxl.append((ww_seq_tmp, sz_ww_seq))

            assert len(sub_total) == sz_ww_seq
            ww_return.append(ww_seq_tmp)
            ww_list.append(ww_tmp)
            # pdb.set_trace()
            _tqdm.update(1)
    # pdb.set_trace()
    if deal_numeric:
        in_kpi = []
        # pdb.set_trace()
        for key in rev_kpi_dic.keys():
            if key in kpi_static_set:
                in_kpi.append(rev_kpi_dic[key])
        if len(in_kpi) < len(rev_kpi_dic):
            print(f"[{len(in_kpi)}] KPI are covered by data: {in_kpi}")
            print(f" [{len(not_in_KPI)}] KPI无法匹配{not_in_KPI}")
        else:
            print("all KPI are covered!")
    return ww_return, kpi_info, sten_that_over_maxl


def extract_num(seq_data_cws):
    '''
        功能：把序列中的数值信息提取出来
        同时过滤 nan 数值
    '''
    num_ref = []
    seq_data_cws_new = []
    for j in range(len(seq_data_cws)):
        num_index = [i for i, x in enumerate(seq_data_cws[j]) if x == '[NUM]']
        # kpi_score = [float(seq_data_cws[i][index+1]) for index in num_index]
        kpi_score = []
        flag = 1
        for index in num_index:
            # if math.isnan(tmp):
            #     pdb.set_trace()
            try:
                tmp = float(seq_data_cws[j][index + 1])
            except:
                # pdb.set_trace()
                flag = 0
                continue
            if math.isnan(tmp):
                flag = 0
            else:
                kpi_score.append(tmp)

        if len(num_index) > 0:
            for index in reversed(num_index):
                seq_data_cws[j].pop(index + 1)
        if flag == 1:
            num_ref.append(kpi_score)
            seq_data_cws_new.append(seq_data_cws[j])
    return seq_data_cws_new, num_ref


def extract_kpi(token_data, kpi_dic, not_in_KPI):
    '''
        功能：把序列中的[KPI]下标范围，[NUM]下标提取出来
        输出格式： [(1,2,4),(5,6,7)]
    '''
    kpi_and_num_info = []
    kpi_type = []
    kpi_index = [i for i, x in enumerate(token_data) if x.lower() == '[kpi]']
    num_index = [i for i, x in enumerate(token_data) if x.lower() == '[num]']
    sz = len(kpi_index)
    assert sz == len(num_index)
    for i in range(sz):
        # (kpi 开始，kpi 结束，NUM token位置)
        # DONE: 添加KPI的类别
        kpi_name = ''.join(token_data[kpi_index[i] + 1: num_index[i] - 1])
        kpi_name_clear = kpi_name.replace('##', '')

        if kpi_name in kpi_dic:
            kpi_id = int(kpi_dic[kpi_name])
        elif kpi_name_clear in kpi_dic:
            kpi_id = int(kpi_dic[kpi_name_clear])
        elif kpi_name_clear in not_in_KPI:
            kpi_id = -1
            not_in_KPI[kpi_name_clear] += 1
        else:
            # 只打印一次
            not_in_KPI[kpi_name_clear] += 1
            kpi_id = -1
            # print(f"{kpi_name_clear} not in KPI dict")

        kpi_info = [kpi_index[i] + 1, num_index[i] - 2, num_index[i], kpi_id]
        kpi_and_num_info.append(kpi_info)
        kpi_type.append(kpi_id)
    # pdb.set_trace()

    return kpi_and_num_info, kpi_type


def kpi_combine(kpi_info, num_ref):
    sz = len(kpi_info)
    assert sz == len(num_ref)
    for i in range(sz):
        for j in range(len(kpi_info[i])):
            kpi_info[i][j].append(num_ref[i][j])
            # pdb.set_trace()
    return kpi_info

# 所有字母小写


def kpi_lower_update(kpi_dic):
    new_dic = {}
    for key in kpi_dic:
        kk = key.lower().split()
        kk = ''.join(kk).strip()
        new_dic[kk] = kpi_dic[key]
    return new_dic


if __name__ == '__main__':
    '''
    功能： 得到 chinese ref 文件，同时刷新训练/测试文件（仅针对序列的文本数据）
    '''
    cfg = cfg()
    cfg.get_args()
    cfgs = cfg.update_train_configs()

    # 路径指定
    domain_file_path = osp.join(cfgs.data_path, 'special_vocab.txt')
    with open(domain_file_path, encoding="utf-8") as f:
        ref = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    tokenizer = BertTokenizer.from_pretrained(osp.join(cfgs.data_root, 'transformer', 'MacBert'), do_lower_case=True)
    seq_data_name = cfgs.seq_data_name
    with open(osp.join(cfgs.data_path, f'{seq_data_name}.json'), "r") as fp:
        seq_data = json.load(fp)
    kpi_dic_name = 'kpi2id'
    with open(osp.join(cfgs.data_path, f'{kpi_dic_name}.json'), "r") as fp:
        kpi_dic = json.load(fp)
    kpi_dic = kpi_lower_update(kpi_dic)
    # 供测试
    random.shuffle(seq_data)
    # seq_data = seq_data[:500]
    print(f"tokenizer size before: {len(tokenizer)}")
    tokenizer, special_token, norm_token = add_special_token(tokenizer)
    special_token = special_token + norm_token

    print(f"tokenizer size after: {len(tokenizer)}")
    print('------------------------ refresh data --------------------------------')
    add_words = refresh_data(ref, cfgs.freq, special_token)

    if not cfgs.read_cws:
        print('------------------------ cws ----------------------------------')
        seq_data_cws, data_size, error_data = cws(seq_data, add_words, cfgs.batch_size)
        print(f'batch size is {cfgs.batch_size}')
        if len(error_data) > 0:
            with open(osp.join(cfgs.data_path, f'{seq_data_name}_error.json'), "w") as fp:
                json.dump(error_data, fp, ensure_ascii=False)
        save_path_cws_orig = osp.join(cfgs.data_path, f'{seq_data_name}_cws_orig.json')
        print("get the new training data! saving...")
        with open(save_path_cws_orig, 'w', ) as fp:
            json.dump(seq_data_cws, fp, ensure_ascii=False)
    else:
        print('------------------------ read ----------------------------------')
        save_path_cws = osp.join(cfgs.data_path, f'{seq_data_name}_cws_orig.json')
        print("get the new training data!")
        with open(save_path_cws, 'r', ) as fp:
            seq_data_cws = json.load(fp)
        data_size = len(seq_data_cws)

    sz_orig = len(seq_data_cws)
    if cfgs.deal_numeric:
        seq_data_cws, num_ref = extract_num(seq_data_cws)
    print(f"过滤了{sz_orig - len(seq_data_cws)}个无效句子")
    data_size = len(seq_data_cws)

    print('---------------------- generate chinese ref ------------------------------')
    chinese_ref, kpi_info, sten_that_over_maxl = generate_chinese_ref(seq_data_cws, special_token, cfgs.deal_numeric, kpi_dic)

    if len(sten_that_over_maxl) > 0:
        print(f"{len(sten_that_over_maxl)} over the 500 len!")
        save_path_max = osp.join(cfgs.data_path, f'{seq_data_name}_max_len_500.json')
        with open(save_path_max, 'w') as fp:
            json.dump(sten_that_over_maxl, fp, ensure_ascii=False)

    if cfgs.deal_numeric:
        print("KPI info combine")
        kpi_ref = kpi_combine(kpi_info, num_ref)
        # pdb.set_trace()
    print('------------------------- match finished ------------------------------')

    # 输出最后训练的时候用于做wwm的分词
    save_path_ref = osp.join(cfgs.data_path, f'{seq_data_name}_chinese_ref.json')
    with open(save_path_ref, 'w') as fp:
        json.dump(chinese_ref, fp, ensure_ascii=False)
    print(f"save chinese_ref done!")

    seq_data_cws_output = []
    for i in range(data_size):
        seq = " ".join(seq_data_cws[i])
        seq_data_cws_output.append(seq)

    save_path_cws = osp.join(cfgs.data_path, f'{seq_data_name}_cws.json')
    print("get the new training data!")
    with open(save_path_cws, 'w', ) as fp:
        json.dump(seq_data_cws_output, fp, ensure_ascii=False)

    print("save seq_data_cws done!")

    if cfgs.deal_numeric:
        kpi_ref_path = osp.join(cfgs.data_path, f'{seq_data_name}_kpi_ref.json')
        with open(kpi_ref_path, 'w', ) as fp:
            json.dump(kpi_ref, fp, ensure_ascii=False)
        print("save num and kpi done!")
