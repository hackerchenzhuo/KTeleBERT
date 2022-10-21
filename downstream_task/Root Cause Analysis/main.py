import os
from fault_pkg import FaultPKG, get_pkg_list, RuleSet
from fault_graph import FaultGraph
import argparse
# from trainer import Trainer
# from trainer_gnnontopo import Trainer
# from trainer_onlyalarmfeat import Trainer
# from trainer_instance import Trainer
from trainer_instance_logistic_multifold import Trainer
from utils import init_dir
import pickle as pkl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='./data/团泊洼实验室数据')
    parser.add_argument('--valid_data_path', default='./data/团泊洼实验室数据测试集')
    parser.add_argument('--map_dict_path', default='./data/map_dict.pkl')
    parser.add_argument('--attr_path', default='./pretrain/fault_attr.json')
    parser.add_argument('--pt_emb_path', default='./pretrain/yht_serialize_withAttribute_MacBert__cls_emb.pt')
    parser.add_argument('--rule_path', default='./rule/rule.txt')
    parser.add_argument('--task_name', default='test')

    # training setting
    parser.add_argument('--num_epoch', default=1000, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--train_bs', default=99, type=int)
    parser.add_argument('--eval_bs', default=64, type=int)
    parser.add_argument('--ent_dim', default=256, type=int)
    parser.add_argument('--nlayer', default=2, type=int)

    # file setting
    parser.add_argument('--state_dir', default='./state', type=str)
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--tb_log_dir', default='./tb_log', type=str)

    # log setting
    parser.add_argument('--log_per_epoch', default=1, type=int)
    parser.add_argument('--valid_per_epoch', default=5, type=int)
    parser.add_argument('--early_stop_patience', default=30, type=int)

    parser.add_argument('--gpu', default='cuda:2', type=str)

    args = parser.parse_args()
    init_dir(args)

    map_dict = pkl.load(open(args.map_dict_path, 'rb'))
    rule_set = RuleSet(args.rule_path)
    # rule_set = None
    train_g_list = [FaultGraph(pkg, map_dict) for pkg in get_pkg_list(args.train_data_path, rule_set)]
    valid_g_list = [FaultGraph(pkg, map_dict) for pkg in get_pkg_list(args.valid_data_path, rule_set)]

    trainer = Trainer(args, train_g_list, valid_g_list, valid_g_list, map_dict)
    trainer.train()