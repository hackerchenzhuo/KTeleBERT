import os
import pdb
from fault_pkg import FaultPKG
from fault_graph import FaultGraph
import argparse
from trainer_instance_logistic_multifold import Trainer
from utils import init_dir
import pickle as pkl
from fault_pkg import RuleSet
from utils import set_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_list', default='./data/rand_pkg_path_yht.pkl')
    parser.add_argument('--map_dict_path', default='./data/map_dict.pkl')
    parser.add_argument('--important_attr_path', default='./pretrain/importantAttributeList.json')

    parser.add_argument('--pt_input_alarm', default='./pretrain/alarm_list.json')
    parser.add_argument('--pt_input_kpi', default='./pretrain/kpi_list.json')
    parser.add_argument('--pt_input', default='./pretrain/fault_attr.json')  # input for pre-training
    parser.add_argument('--pt_emb_path', default='./pretrain/yht_serialize_withoutAttr_emb_Pre_train_0926_v41_seed42_epoch12_last_avg.pt')

    parser.add_argument('--num_fold', default=5, type=int)
    parser.add_argument('--rule_path', default='./rule/rule.txt')
    parser.add_argument('--task_group_name', default='test')
    parser.add_argument('--task_name', default=None)
    parser.add_argument('--use_rule', default="count")
    parser.add_argument('--withAttr', default='True')
    parser.add_argument('--onlyName', default='True')

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

    parser.add_argument('--gpu', default='cuda:0', type=str)

    args = parser.parse_args()
    init_dir(args)

    if "onlyname" in args.pt_emb_path:
        args.withAttr = 'onlyname'
    elif "withAttr" in args.pt_emb_path:
        args.withAttr = 'True'
    else:
        args.withAttr = 'False'
    
    if ".pt" in args.pt_emb_path:

        set_seed(3427)

        data_path_list = pkl.load(open(args.data_path_list, 'rb'))
        fold_list = []
        num_data_in_fold = len(data_path_list) // args.num_fold

        # multi fold training and testing
        for i in range(0, len(data_path_list), num_data_in_fold):
            if i+num_data_in_fold > len(data_path_list):
                fold_list[-1].extend(data_path_list[i:])
            else:
                fold_list.append(data_path_list[i: i+num_data_in_fold])

        rule_set = RuleSet(args.rule_path)
        fold_g_list = [[FaultGraph(FaultPKG(args, p, rule_set)) for p in fold] for fold in fold_list]
        fold_g_list = fold_g_list + fold_g_list

        fold_run_res = 'Task Name\tMR\tH@1\tH@3\tH@5\n'
        avg_res = {'MR': 0, 'H@1': 0, 'H@3': 0, 'H@5': 0}
        for i in range(args.num_fold):
            valid_fault_g = fold_g_list[i]
            test_fault_g = fold_g_list[i + 1]
            train_fault_g = []
            for j in range(i+2, i+args.num_fold):
                train_fault_g.extend(fold_g_list[j])

            args.task_name = args.task_group_name + f'_run{i}'
            trainer = Trainer(args, train_fault_g, valid_fault_g, test_fault_g)
            res = trainer.train()
            fold_run_res += f"{args.task_name}\t{res['MR']:.4f}\t{res['H@1']:.4f}\t{res['H@3']:.4f}\t{res['H@5']:.4f}\n"
            for k, v in avg_res.items():
                avg_res[k] += res[k]

        for k, v in avg_res.items():
            avg_res[k] = v / len(fold_list)

        fold_run_res += f"Avg Res\t{avg_res['MR']:.4f}\t{avg_res['H@1']:.4f}\t{avg_res['H@3']:.4f}\t{avg_res['H@5']:.4f}\n"

        with open(os.path.join(args.log_dir, f'{args.task_group_name}.log'), 'w') as f:
            f.write(fold_run_res)
