from torch.utils.tensorboard import SummaryWriter
from utils import Log
import json
import os
from torch.utils.data import DataLoader
from dataset import FaultGraphDataset
import torch.nn as nn
import torch
from torch import optim
from fault_pkg import alarm_name2idx, nftype_name2idx, label_name2idx, get_pkg_list
import numpy as np
import dgl
# from graph_classifier import GraphClassifier
from gnn import GNN, DGLGCN
from ent_init import EntInit, EntInitAlarmFeat, MLP
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

        self.train_pkg_list = get_pkg_list(args.train_data_path)
        self.valid_pkg_list = get_pkg_list(args.valid_data_path)

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
        self.ent_init = EntInit(args).to(self.args.gpu)
        # self.gnn = GNN(args, len(alarm_name2idx), 1024, 512, nlayer=args.nlayer).to(self.args.gpu)
        self.gnn = GNN(args, 768, 1024, 512, nlayer=args.nlayer).to(self.args.gpu)
        # self.gnn = GNN(args, 10, 20, 10, nlayer=args.nlayer).to(self.args.gpu)
        # self.gnn = DGLGCN(args, node_dim=args.ent_dim, nlayer=args.nlayer).to(self.args.gpu)
        # self.graph_classifier = GraphClassifier(args).to(self.args.gpu)
        self.graph_classifier = MLP(512, 128, len(label_name2idx)).to(self.args.gpu)
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
        self.loss_func = nn.CrossEntropyLoss()

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
        state = torch.load(os.path.josin(self.state_path, self.name + '.best'), map_location=self.args.gpu)
        self.ent_init.load_state_dict(state['ent_init'])
        self.gnn.load_state_dict(state['neighbor_agg'])
        self.graph_classifier.load_state_dict(state['graph_classifier'])

    def get_ent_emb(self, g):
        ent_emb = self.ent_init(g)

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
            self.ent_init.train()
            self.gnn.train()
            self.graph_classifier.train()

            train_logit = []
            train_target = []
            batch_losses = []
            for batch in self.train_loader:
                batch_g = dgl.batch([b[0].g for b in batch]).to(self.args.gpu)

                ent_emb = self.get_ent_emb(batch_g)
                ent_emb = self.gnn(batch_g, ent_emb)
                batch_ent_emb = self.split_emb(ent_emb, batch_g.batch_num_nodes().tolist())

                batch_graph_repr = []
                batch_target = []
                for batch_i, batch_data in enumerate(batch):
                    pool_emb = batch_ent_emb[batch_i]
                    graph_repr = torch.max(pool_emb, dim=0)[0]
                    # graph_repr = self.graph_classifier(graph_repr)
                    batch_graph_repr.append(graph_repr)
                    batch_target.append(batch_data[1])

                batch_graph_repr = self.graph_classifier(torch.stack(batch_graph_repr))
                batch_logit = torch.max(batch_graph_repr, dim=-1)[1]
                batch_target = torch.LongTensor(batch_target).to(self.args.gpu)

                train_logit.append(batch_logit)
                train_target.append(batch_target)

                batch_loss = self.loss_func(batch_graph_repr, batch_target)

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                batch_losses.append(batch_loss.item())

            train_logit = torch.cat(train_logit)
            train_target = torch.cat(train_target)
            acc = sum(train_target == train_logit) / train_target.shape[0]

            epoch_loss = sum(batch_losses) / len(batch_losses)

            self.write_training_loss(epoch_loss, e)
            self.write_training_acc(acc, e)

            if e % self.log_per_epoch == 0:
                self.logger.info(f'step: {e} | loss: {epoch_loss:.4f} | acc:{acc:.4f}')

            if e % self.valid_per_epoch == 0 and e != 0:
                valid_acc = self.eval(self.valid_loader)
                self.write_valid_acc(valid_acc, e)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_epoch = e
                    self.logger.info('best model | acc {:.4f}'.format(best_acc))
                    self.save_checkpoint(e, self.get_curr_state())
                    bad_count = 0
                else:
                    bad_count += 1
                    self.logger.info(f'bad count {bad_count} | acc {valid_acc:.4f}; best: epoch {best_epoch}, acc {best_acc:.4f}')

    def eval(self, data_loader):
        self.ent_init.eval()
        self.gnn.eval()
        self.graph_classifier.eval()

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
                pool_emb = batch_ent_emb[batch_i]
                graph_repr = torch.mean(pool_emb, dim=0)
                batch_graph_repr.append(graph_repr)
                batch_target.append(batch_data[1])

            batch_graph_repr = self.graph_classifier(torch.stack(batch_graph_repr))
            batch_target = torch.LongTensor(batch_target).to(self.args.gpu)
            batch_logit = torch.max(batch_graph_repr, dim=-1)[1]
            eval_logit.append(batch_logit)
            eval_target.append(batch_target)

        eval_logit = torch.cat(eval_logit)
        eval_target = torch.cat(eval_target)

        acc = sum(eval_target == eval_logit) / eval_target.shape[0]

        return acc


