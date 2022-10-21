from torch.utils.tensorboard import SummaryWriter
from utils import Log
import json
import os
from torch.utils.data import DataLoader
from dataset import FaultGraphDataset
import torch.nn as nn
import torch
from torch import optim
from fault_pkg import alarm_name2idx, nftype_name2idx, label_name2idx, get_pkg_list, RuleSet
import numpy as np
import dgl
from graph_classifier import NodeClassifier
from gnn import GNN, DGLGCN
from ent_init import EntInit, EntInitAlarmFeat, MLP, EntInitWithAttr
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.name = args.task_name
        self.writer = SummaryWriter(os.path.join(args.tb_log_dir, self.name))
        self.logger = Log(args.log_dir, self.name).get_logger()
        self.logger.info(json.dumps({k: v for k, v in vars(args).items()}))

        # state dir
        self.state_path = os.path.join(args.state_dir, self.name)
        if not os.path.exists(self.state_path):
            os.makedirs(self.state_path)

        self.train_bs = args.train_bs
        self.valid_bs = args.valid_bs
        self.num_epoch = args.num_epoch
        self.lr = args.lr
        self.log_per_epoch = args.log_per_epoch
        self.valid_per_epoch = args.valid_per_epoch
        self.early_stop_patience = args.early_stop_patience

        self.rule_set = RuleSet(args)
        self.train_pkg_list = get_pkg_list(args.train_data_path, self.rule_set)
        self.valid_pkg_list = get_pkg_list(args.valid_data_path, self.rule_set)

        self.train_loader = DataLoader(FaultGraphDataset(args, self.train_pkg_list),
                                       batch_size=self.train_bs,
                                       shuffle=True,
                                       collate_fn=FaultGraphDataset.collate_fn,
                                       drop_last=False)

        self.valid_loader = DataLoader(FaultGraphDataset(args, self.valid_pkg_list),
                                       batch_size=self.valid_bs,
                                       shuffle=False,
                                       collate_fn=FaultGraphDataset.collate_fn,
                                       drop_last=False)

        # GNN
        # self.ent_init = EntInitAlarmFeat(args).to(self.args.gpu)
        # self.ent_init = EntInitWithAttr(args).to(self.args.gpu)
        self.ent_init = EntInit(args).to(self.args.gpu)
        # self.gnn = GNN(args, len(alarm_name2idx), 1024, 512, nlayer=args.nlayer).to(self.args.gpu)
        # self.gnn = GNN(args, 200, 500, 200, nlayer=args.nlayer).to(self.args.gpu)
        self.gnn = GNN(args, 768, 1024, 512, nlayer=args.nlayer).to(self.args.gpu)
        # self.gnn = GNN(args, 10, 20, 10, nlayer=args.nlayer).to(self.args.gpu)
        # self.gnn = DGLGCN(args, node_dim=args.ent_dim, nlayer=args.nlayer).to(self.args.gpu)
        # self.graph_classifier = GraphClassifier(args).to(self.args.gpu)
        # self.graph_classifier = NodeClassifier(200, 50, 1).to(self.args.gpu)
        self.graph_classifier = NodeClassifier(512, 128, 1).to(self.args.gpu)
        # self.graph_classifier = MLP(10, 10, len(label_name2idx)).to(self.args.gpu)

        # init embedding

        self.alarm_emb = nn.Parameter(torch.Tensor(len(alarm_name2idx), args.ent_dim).to(args.gpu))
        nn.init.xavier_uniform_(self.alarm_emb, gain=nn.init.calculate_gain('relu'))

        self.nftype_emb = nn.Parameter(torch.Tensor(len(nftype_name2idx), args.ent_dim).to(args.gpu))
        nn.init.xavier_uniform_(self.nftype_emb, gain=nn.init.calculate_gain('relu'))

        # optimizer
        self.optimizer = optim.Adam(
                list(self.ent_init.parameters()) +
                list(self.gnn.parameters()) +
                list(self.graph_classifier.parameters()),
                lr=self.lr)
            # weight_decay=1e-5)

        # loss function
        # label_num = torch.zeros(len(label_name2idx))
        # for p in train_pkg_list:
        #     p_lable_idx = label_name2idx[p.label.fault_type]
        #     label_num[p_lable_idx] += 1
        # label_weight = 1/label_num
        # label_weight[label_num == 0] = 0
        # label_weight = F.softmax(label_weight/0.2, dim=0).to(args.gpu)
        # self.loss_func = nn.CrossEntropyLoss(weight=label_weight)
        # self.loss_func = nn.CrossEntropyLoss()

    def loss_func(self, score, label):
        l = torch.log(1 + torch.exp(-label*score))
        l = torch.mean(l)
        return l

    def write_valid_acc(self, acc, e):
        self.writer.add_scalar("validation/acc", acc, e)

    def write_training_loss(self, loss, step):
        self.writer.add_scalar("training/loss", loss, step)

    def write_training_acc(self, acc, step):
        self.writer.add_scalar("training/acc", acc, step)

    def save_checkpoint(self, e, state):
        # delete previous checkpoint
        for filename in os.listdir(self.state_path):
            if self.name in filename.split('.') and os.path.isfile(os.path.join(self.state_path, filename)):
                os.remove(os.path.join(self.state_path, filename))
        # save checkpoint
        torch.save(state, os.path.join(self.args.state_dir, self.name,
                                       self.name + '.' + str(e) + '.ckpt'))

    def save_model(self, best_step):
        os.rename(os.path.join(self.state_path, self.name + '.' + str(best_step) + '.ckpt'),
                  os.path.join(self.state_path, self.name + '.best'))

    def get_curr_state(self):
        state = {
                 'ent_init': self.ent_init.state_dict(),
                 'gnn': self.gnn.state_dict(),
                 'graph_classifier': self.graph_classifier.state_dict(),
        }
        return state

    def before_test_load(self):
        state = torch.load(os.path.join(self.state_path, self.name + '.best'), map_location=self.args.gpu)
        self.ent_init.load_state_dict(state['ent_init'])
        self.gnn.load_state_dict(state['neighbor_agg'])
        self.graph_classifier.load_state_dict(state['graph_classifier'])

    def get_ent_emb(self, g, batch_ent_alarm_idx):
        ent_emb = self.ent_init(g, batch_ent_alarm_idx)

        return ent_emb

    def split_emb(self, emb, split_list):
        split_list = [np.sum(split_list[0: i], dtype=np.int) for i in range(len(split_list) + 1)]
        emb_split = [emb[split_list[i]: split_list[i + 1]] for i in range(len(split_list) - 1)]
        return emb_split

    def train(self):
        best_epoch = 0
        best_rank = 100
        bad_count = 0
        self.logger.info('start training')

        for e in range(1, self.num_epoch + 1):
            self.ent_init.train()
            self.gnn.train()
            self.graph_classifier.train()

            train_rank = []
            batch_losses = []
            for batch in self.train_loader:
                batch_g = dgl.batch([b[0].g for b in batch]).to(self.args.gpu)
                batch_ent_alarm_idx = []
                for b in batch:
                    batch_ent_alarm_idx.extend(b[0].ent_alarm_id_list)

                ent_emb = self.get_ent_emb(batch_g, batch_ent_alarm_idx)
                ent_emb = self.gnn(batch_g, ent_emb)

                # loss
                batch_ins_score = self.graph_classifier(ent_emb).reshape(-1)
                batch_ins_label = torch.cat([torch.LongTensor(batch[2]) for batch in batch]).to(self.args.gpu)

                batch_loss = self.loss_func(batch_ins_score, batch_ins_label)

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                batch_losses.append(batch_loss.item())

                # rank
                split_ins_score = self.split_emb(batch_ins_score, batch_g.batch_num_nodes().tolist())
                for batch_i, batch_data in enumerate(batch):
                    split_ins_label = batch_data[2]
                    rank = self.get_rank(split_ins_score[batch_i],
                                         torch.LongTensor(split_ins_label))
                    train_rank.append(rank)

            train_res = {'MR': np.mean(train_rank), 'H@1': np.sum(np.array(train_rank) == 1) / len(train_rank),
                         'H@3': np.sum(np.array(train_rank) <= 3) / len(train_rank),
                         'H@5': np.sum(np.array(train_rank) <= 5) / len(train_rank)}

            epoch_loss = sum(batch_losses) / len(batch_losses)

            self.write_training_loss(epoch_loss, e)
            self.write_training_acc(rank, e)

            if e % self.log_per_epoch == 0:
                self.logger.info(f"step: {e} | loss: {epoch_loss:.4f} | MR: {train_res['MR']:.4f}, "
                                 f"H@1: {train_res['H@1']:.4f}, H@3: {train_res['H@3']:.4f}, H@5: {train_res['H@5']:.4f}")

            if e % self.valid_per_epoch == 0 and e != 0:
                valid_res = self.eval(self.valid_loader)
                valid_rank = valid_res['MR']
                self.write_valid_acc(valid_rank, e)
                if valid_rank < best_rank:
                    best_rank = valid_rank
                    best_epoch = e
                    self.logger.info(f"best model | MR: {valid_res['MR']:.4f}, H@1: {valid_res['H@1']:.4f}, "
                                     f"H@3: {valid_res['H@3']:.4f}, H@5: {valid_res['H@5']:.4f}")
                    self.save_checkpoint(e, self.get_curr_state())
                    bad_count = 0
                else:
                    bad_count += 1
                    self.logger.info(f"bad count: {bad_count} | MR: {valid_res['MR']:.4f}, H@1: {valid_res['H@1']:.4f}, "
                                     f"H@3: {valid_res['H@3']:.4f}, H@5: {valid_res['H@5']:.4f} | "
                                     f"best epoch: {best_epoch}, MR: {best_rank:.4f}")

    def get_rank(self, score_list, label_list):
        sort_idx = torch.argsort(score_list, descending=True)
        rank = np.argwhere(label_list[sort_idx].numpy() == 1)[0].item() + 1

        return rank

    def eval(self, data_loader):
        self.ent_init.eval()
        self.gnn.eval()
        self.graph_classifier.eval()

        eval_rank = []
        for batch in data_loader:
            batch_g = dgl.batch([b[0].g for b in batch]).to(self.args.gpu)
            batch_ent_alarm_idx = []
            for b in batch:
                batch_ent_alarm_idx.extend(b[0].ent_alarm_id_list)

            ent_emb = self.get_ent_emb(batch_g, batch_ent_alarm_idx)
            ent_emb = self.gnn(batch_g, ent_emb)

            batch_ins_score = self.graph_classifier(ent_emb).reshape(-1)

            # rank
            split_ins_score = self.split_emb(batch_ins_score, batch_g.batch_num_nodes().tolist())
            for batch_i, batch_data in enumerate(batch):
                split_ins_label = batch_data[2]
                rank = self.get_rank(split_ins_score[batch_i],
                                     torch.LongTensor(split_ins_label))
                eval_rank.append(rank)

        eval_res = {'MR': np.mean(eval_rank), 'H@1': np.sum(np.array(eval_rank) == 1) / len(eval_rank),
                    'H@3': np.sum(np.array(eval_rank) <= 3) / len(eval_rank),
                    'H@5': np.sum(np.array(eval_rank) <= 5) / len(eval_rank)}

        return eval_res


