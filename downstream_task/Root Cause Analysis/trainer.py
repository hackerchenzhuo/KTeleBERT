from torch.utils.tensorboard import SummaryWriter
from utils import Log
import json
import os
from torch.utils.data import DataLoader
from dataset import FaultGraphDataset
import torch.nn as nn
import torch
from torch import optim
from gnn import GNN
from fault_pkg import *
import numpy as np
import dgl
from graph_classifier import GraphClassifier


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

        self.train_loader = DataLoader(FaultGraphDataset(args, args.train_data_path),
                                       batch_size=self.train_bs,
                                       shuffle=True,
                                       collate_fn=FaultGraphDataset.collate_fn)

        self.valid_loader = DataLoader(FaultGraphDataset(args, args.valid_data_path),
                                       batch_size=self.valid_bs,
                                       shuffle=True,
                                       collate_fn=FaultGraphDataset.collate_fn)

        # GNN
        self.gnn = GNN(args, node_dim=args.ent_dim, nlayer=args.nlayer).to(self.args.gpu)
        self.graph_classifire = GraphClassifier(args).to(self.args.gpu)

        # init embedding
        global alarm_name2idx
        global nftype_name2idx

        self.alarm_emb = nn.Parameter(torch.Tensor(len(alarm_name2idx), args.ent_dim).to(args.gpu))
        nn.init.xavier_uniform_(self.alarm_emb, gain=nn.init.calculate_gain('relu'))

        self.nftype_emb = nn.Parameter(torch.Tensor(len(nftype_name2idx), args.ent_dim).to(args.gpu))
        nn.init.xavier_uniform_(self.nftype_emb, gain=nn.init.calculate_gain('relu'))

        # optimizer
        self.optimizer = optim.Adam(
                list(self.gnn.parameters()) +
                list(self.graph_classifire.parameters()) +
                [self.alarm_emb, self.nftype_emb],
                lr=self.lr)

        # loss function
        self.loss_func = nn.CrossEntropyLoss()

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
                 'gnn': self.gnn.state_dict(),
                 'graph_classifire': self.graph_classifire.state_dict(),
                 'alarm_emb': self.alarm_emb,
                 'nftype_emb': self.nftype_emb
        }
        return state

    def before_test_load(self):
        state = torch.load(os.path.join(self.state_path, self.name + '.best'), map_location=self.args.gpu)
        self.gnn.load_state_dict(state['neighbor_agg'])
        self.graph_classifire.load_state_dict(state['graph_classifire'])
        self.alarm_emb = state['alarm_emb']
        self.nftype_emb = state['nftype_emb']

    def get_ent_emb(self, g):
        ent_emb = torch.zeros(g.num_nodes(), self.args.ent_dim).to(self.args.gpu)
        is_alarm_idx = (g.ndata['alarm_idx'] != -1)
        is_nftype_idx = (g.ndata['nftype_idx'] != -1)

        ent_emb[is_alarm_idx] = self.alarm_emb[g.ndata['alarm_idx'][is_alarm_idx]]
        ent_emb[is_nftype_idx] = self.nftype_emb[g.ndata['nftype_idx'][is_nftype_idx]]

        return ent_emb

    def split_emb(self, emb, split_list):
        split_list = [np.sum(split_list[0: i], dtype=np.int) for i in range(len(split_list) + 1)]
        emb_split = [emb[split_list[i]: split_list[i + 1]] for i in range(len(split_list) - 1)]
        return emb_split

    def train(self):
        best_epoch = 0
        best_acc = 0
        bad_count = 0
        self.logger.info('start training')

        for e in range(1, self.num_epoch + 1):
            batch_losses = []
            for batch in self.train_loader:
                batch_g = dgl.batch([b[0].g for b in batch]).to(self.args.gpu)

                ent_emb = self.get_ent_emb(batch_g)
                ent_emb = self.gnn(batch_g, ent_emb)
                batch_ent_emb = self.split_emb(ent_emb, batch_g.batch_num_nodes().tolist())

                batch_graph_repr = []
                batch_target = []
                for batch_i, batch_data in enumerate(batch):
                    pool_idx = (batch_data[0].g.ndata['nftype_idx'] != -1)
                    pool_emb = batch_ent_emb[batch_i][pool_idx]
                    graph_repr = torch.max(pool_emb, dim=0)[0]
                    graph_repr = self.graph_classifire(graph_repr)
                    batch_graph_repr.append(graph_repr)
                    batch_target.append(batch_data[1])

                batch_graph_repr = torch.stack(batch_graph_repr)
                batch_target = torch.LongTensor(batch_target).to(self.args.gpu)

                batch_loss = self.loss_func(batch_graph_repr, batch_target)

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                batch_losses.append(batch_loss.item())

            batch_losses = sum(batch_losses) / len(batch_losses)

            if e % self.log_per_epoch == 0:
                self.logger.info('step: {} | loss: {:.4f}'.format(e, batch_losses))

            if e % self.valid_per_epoch == 0 and e != 0:
                valid_acc = self.eval(self.valid_loader)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_epoch = e
                    self.logger.info('best model | acc {:.4f}'.format(best_acc))
                    self.save_checkpoint(e, self.get_curr_state())
                    bad_count = 0
                else:
                    bad_count += 1
                    self.logger.info('best model is at epoch {0}, acc {1:.4f}, bad count {2}'.format(
                        best_epoch, best_acc, bad_count))

    def eval(self, data_loader):
        eval_logit = []
        eval_target = []
        for batch in data_loader:
            batch_g = dgl.batch([b[0].g for b in batch]).to(self.args.gpu)

            ent_emb = self.get_ent_emb(batch_g)
            ent_emb = self.gnn(batch_g, ent_emb)
            batch_ent_emb = self.split_emb(ent_emb, batch_g.batch_num_nodes().tolist())

            batch_graph_repr = []
            batch_target = []
            for batch_i, batch_data in enumerate(batch):
                pool_idx = (batch_data[0].g.ndata['nftype_idx'] != -1)
                pool_emb = batch_ent_emb[batch_i][pool_idx]
                graph_repr = torch.max(pool_emb, dim=0)[0]
                graph_repr = self.graph_classifire(graph_repr)
                batch_graph_repr.append(graph_repr)
                batch_target.append(batch_data[1])

            batch_graph_repr = torch.stack(batch_graph_repr)
            batch_target = torch.LongTensor(batch_target).to(self.args.gpu)
            batch_logit = torch.max(batch_graph_repr, dim=-1)[1]
            eval_logit.append(batch_logit)
            eval_target.append(batch_target)

        eval_logit = torch.cat(eval_logit)
        eval_target = torch.cat(eval_target)

        acc = sum(eval_target == eval_logit) / eval_target.shape[0]

        return acc


