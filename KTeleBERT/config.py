import os.path as osp
import numpy as np
import random
import torch
from easydict import EasyDict as edict
import argparse


LAYER_MAPPING = {
    0: 'od_layer_0',
    1: 'od_layer_1',
    2: 'od_layer_2',
}


class cfg():
    def __init__(self):
        self.this_dir = osp.dirname(__file__)
        # change
        self.data_root = osp.abspath(osp.join(self.this_dir, '..', '..', 'data', ''))

        # TODO: add some static variable  (The frequency of change is low)

    def get_args(self):
        parser = argparse.ArgumentParser()
        # ------------ base ------------
        parser.add_argument('--train_strategy', default=1, type=int)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--batch_size_ke', default=14, type=int)
        parser.add_argument('--batch_size_od', default=8, type=int)
        parser.add_argument('--batch_size_ad', default=32, type=int)

        parser.add_argument('--epoch', default=15, type=int)
        parser.add_argument("--save_model", default=1, type=int, choices=[0, 1])
        # 用transformer的 save_pretrain 方式保存
        parser.add_argument("--save_pretrain", default=0, type=int, choices=[0, 1])
        parser.add_argument("--from_pretrain", default=0, type=int, choices=[0, 1])

        # torthlight
        parser.add_argument("--no_tensorboard", default=False, action="store_true")
        parser.add_argument("--exp_name", default="huawei_exp", type=str, help="Experiment name")
        parser.add_argument("--dump_path", default="dump/", type=str, help="Experiment dump path")
        parser.add_argument("--exp_id", default="ke256_raekt_ernie2_bs20_p3_c3_5e-6", type=str, help="Experiment ID")
        # or 3407
        parser.add_argument("--random_seed", default=42, type=int)
        # 数据参数
        parser.add_argument("--data_path", default="huawei", type=str, help="Experiment path")
        parser.add_argument('--train_ratio', default=1, type=float, help='ratio for train/test')
        parser.add_argument("--seq_data_name", default='Seq_data_base', type=str, help="seq_data 名字")
        parser.add_argument("--kg_data_name", default='KG_data_base_rule', type=str, help="kg_data 名字")
        parser.add_argument("--order_data_name", default='event_order_data', type=str, help="order_data 名字")
        # TODO: add some dynamic variable
        parser.add_argument("--model_name", default="MacBert", type=str, help="model name")

        # ------------ 训练阶段 ------------
        parser.add_argument("--scheduler", default="cos", type=str, choices=["linear", "cos"])
        parser.add_argument("--optim", default="adamw", type=str)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument('--workers', type=int, default=8)
        parser.add_argument('--accumulation_steps', type=int, default=6)
        parser.add_argument('--accumulation_steps_ke', type=int, default=6)
        parser.add_argument('--accumulation_steps_ad', type=int, default=6)
        parser.add_argument('--accumulation_steps_od', type=int, default=6)
        parser.add_argument("--train_together", default=0, type=int)


        # 3e-5
        parser.add_argument('--lr', type=float, default=1e-5)
        # 逐层学习率衰减
        parser.add_argument("--LLRD", default=0, type=int, choices=[0, 1])
        # parser.add_argument('--margin', default=9.0, type=float, help='The fixed margin in loss function. ')
        # parser.add_argument('--emb_dim', default=1000, type=int, help='The embedding dimension in KGE model.')
        # parser.add_argument('--adv_temp', default=1.0, type=float, help='The temperature of sampling in self-adversarial negative sampling.')
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        parser.add_argument('--scheduler_steps', type=int, default=None,
                            help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
        parser.add_argument('--eval_step', default=100, type=int, help='evaluate each n step')

        # ------------ PLM ------------
        parser.add_argument('--maxlength', type=int, default=200)
        parser.add_argument('--mlm_probability', type=float, default=0.15)
        parser.add_argument('--final_mlm_probability', type=float, default=0.4)
        parser.add_argument('--mlm_probability_increase', type=str, default="curve", choices=["linear", "curve"])
        parser.add_argument("--mask_stratege", default="rand", type=str, choices=["rand", "wwm", "domain"])
        # 前n个epoch 用rand，后面用wwm. multi-stage knowledge masking strategy
        parser.add_argument("--ernie_stratege", default=-1, type=int)
        # 用mlm任务进行训练,默认使用chinese_ref且添加新的special word
        parser.add_argument("--use_mlm_task", default=1, type=int, choices=[0, 1])
        # 添加新的special word
        parser.add_argument("--add_special_word", default=1, type=int, choices=[0, 1])
        # freeze
        parser.add_argument("--freeze_layer", default=0, type=int, choices=[0, 1, 2, 3, 4])
        # 是否mask 特殊token
        parser.add_argument("--special_token_mask", default=0, type=int, choices=[0, 1])
        parser.add_argument("--emb_init", default=1, type=int, choices=[0, 1])
        parser.add_argument("--cls_head_init", default=1, type=int, choices=[0, 1]) 
        # 是否使用自适应权重
        parser.add_argument("--use_awl", default=1, type=int, choices=[0, 1])
        parser.add_argument("--mask_loss_scale", default=1.0, type=float)
        
        # ------------ KGE ------------
        parser.add_argument('--ke_norm', type=int, default=1)
        parser.add_argument('--ke_dim', type=int, default=768)
        parser.add_argument('--ke_margin', type=float, default=1.0)
        parser.add_argument('--neg_num', type=int, default=10)
        parser.add_argument('--adv_temp', type=float, default=1.0, help='The temperature of sampling in self-adversarial negative sampling.')
        # 5e-4
        parser.add_argument('--ke_lr', type=float, default=3e-5)
        parser.add_argument('--only_ke_loss', type=int, default=0)

        # ------------ 数值embedding相关 ------------
        parser.add_argument('--use_NumEmb', type=int, default=1)
        parser.add_argument("--contrastive_loss", default=1, type=int, choices=[0, 1])
        parser.add_argument("--l_layers", default=2, type=int)
        parser.add_argument('--use_kpi_loss', type=int, default=1)

        # ------------ 测试阶段 ------------
        parser.add_argument("--only_test", default=0, type=int, choices=[0, 1])
        parser.add_argument("--mask_test", default=0, type=int, choices=[0, 1])
        parser.add_argument("--embed_gen", default=0, type=int, choices=[0, 1])
        parser.add_argument("--ke_test", default=0, type=int, choices=[0, 1])
        # -1: 测全集
        parser.add_argument("--ke_test_num", default=-1, type=int)
        parser.add_argument("--path_gen", default="", type=str)

        # ------------ 时序阶段 ------------
        # 1：预训练
        # 2：时序 finetune
        # 3. 异常检测 finetune + 时序, 且是迭代的
        # 是否加载od模型
        parser.add_argument("--order_load", default=0, type=int)
        parser.add_argument("--order_num", default=2, type=int)
        parser.add_argument("--od_type", default='linear_cat', type=str, choices=['linear_cat', 'vertical_attention'])
        parser.add_argument("--eps", default=0.2, type=float, help='label smoothing..')
        parser.add_argument("--num_od_layer", default=0, type=int)
        parser.add_argument("--plm_emb_type", default='cls', type=str, choices=['cls', 'last_avg'])
        parser.add_argument("--order_test_name", default='', type=str)
        parser.add_argument("--order_threshold", default=0.5, type=float)
        # ------------ 并行训练 ------------
        # 是否并行
        parser.add_argument('--rank', type=int, default=0, help='rank to dist')
        parser.add_argument('--dist', type=int, default=0, help='whether to dist')
        # 不要改该参数，系统会自动分配
        parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
        # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
        parser.add_argument('--world-size', default=4, type=int,
                            help='number of distributed processes')
        parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
        parser.add_argument("--local_rank", default=-1, type=int)
        self.cfg = parser.parse_args()

    def update_train_configs(self):
        # add some constraint for parameters
        # e.g. cannot save and test at the same time
        # 修正默认参数
        # TODO: 测试逻辑有问题需要修改
        if len(self.cfg.order_test_name) > 0:
            self.cfg.save_model = 0
            if len(self.cfg.order_test_name) == 0:
                self.cfg.train_ratio = min(0.8, self.cfg.train_ratio)
            # 自适应载入文件名
            else:
                print("od test ... ")
                self.cfg.train_strategy == 5
                self.cfg.plm_emb_type = 'last_avg' if 'last_avg' in self.cfg.model_name else 'cls'
                for key in LAYER_MAPPING.keys():
                    if LAYER_MAPPING[key] in self.cfg.model_name:
                        self.cfg.num_od_layer = key
                self.cfg.order_test_name = osp.join('downstream_task', f'{self.cfg.order_test_name}')

        if self.cfg.mask_test or self.cfg.embed_gen or self.cfg.ke_test or len(self.cfg.order_test_name) > 0:
            assert len(self.cfg.model_name) > 0
            self.cfg.only_test = 1
        if self.cfg.only_test == 1:
            self.save_model = 0
            self.save_pretrain = 0

        # TODO: update some dynamic variable
        self.cfg.data_root = self.data_root
        self.cfg.data_path = osp.join(self.data_root, self.cfg.data_path)
        self.cfg.plm_path = osp.join(self.data_root, 'transformer')
        self.cfg.dump_path = osp.join(self.cfg.data_path, self.cfg.dump_path)
        # bs 控制尽量在32
        
        # 自适应权重的数量
        self.cfg.awl_num = 1
        # ------------ 数值embedding相关 ------------
        self.cfg.hidden_size = 768
        self.cfg.num_attention_heads = 8
        self.cfg.hidden_dropout_prob = 0.1
        self.cfg.num_kpi = 304
        self.cfg.specail_emb_path = None
        if self.cfg.emb_init:
            self.cfg.specail_emb_path = osp.join(self.cfg.data_path, 'added_vocab_embedding.pt')
        
        # ------------- 多任务学习相关 -------------
        # 四个阶段
        self.cfg.mask_epoch, self.cfg.ke_epoch, self.cfg.ad_epoch, self.cfg.od_epoch = None, None, None, None
        # 触发多任务 学习
        if self.cfg.train_strategy > 1:
            self.cfg.mask_epoch = [0, 1, 1, 1, 0]
            self.cfg.ke_epoch = [4, 3, 2, 2, 0]
            if self.cfg.only_ke_loss:
                self.cfg.mask_epoch = [0, 0, 0, 0, 0]
            self.cfg.epoch = sum(self.cfg.mask_epoch) + sum(self.cfg.ke_epoch)
            if self.cfg.train_strategy > 2:
                self.cfg.ad_epoch = [0, 6, 3, 1, 0]
                self.cfg.epoch += sum(self.cfg.ad_epoch)
                if self.cfg.train_strategy > 3 and not self.cfg.only_ke_loss:
                    self.cfg.od_epoch = [0, 0, 9, 1, 0]
                    # self.cfg.mask_epoch[3] = 1
                    self.cfg.epoch += sum(self.cfg.od_epoch)
            self.cfg.epoch_matrix = []
            for epochs in [self.cfg.mask_epoch, self.cfg.ke_epoch, self.cfg.ad_epoch, self.cfg.od_epoch]:
                if epochs is not None:
                    self.cfg.epoch_matrix.append(epochs)
            if self.cfg.train_together:
                # loss 直接相加，训练epoch就是mask的epoch
                self.cfg.epoch = sum(self.cfg.mask_epoch)
                self.cfg.batch_size = int((self.cfg.batch_size - 16) / self.cfg.train_strategy)
                self.cfg.batch_size_ke = int(self.cfg.batch_size_ke / self.cfg.train_strategy) - 2
                self.cfg.batch_size_ad = int(self.cfg.batch_size_ad / self.cfg.train_strategy) - 1
                self.cfg.batch_size_od = int(self.cfg.batch_size_od / self.cfg.train_strategy) - 1
                self.cfg.accumulation_steps = (self.cfg.accumulation_steps-1) * self.cfg.train_strategy

        self.cfg.neg_num = max(min(self.cfg.neg_num, self.cfg.batch_size_ke - 3), 1)

        self.cfg.accumulation_steps_dict = {0:self.cfg.accumulation_steps, 1:self.cfg.accumulation_steps_ke, 2:self.cfg.accumulation_steps_ad, 3:self.cfg.accumulation_steps_od }

        # 使用数值embedding也必须添加新词因为位置信息和tokenizer绑定
        if self.cfg.use_mlm_task or self.cfg.use_NumEmb:
            assert self.cfg.add_special_word == 1

        if self.cfg.use_NumEmb:
            self.cfg.awl_num += 1

        return self.cfg
