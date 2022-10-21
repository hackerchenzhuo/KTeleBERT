from torch.utils.tensorboard import SummaryWriter
from utils import Log
import json
import os
from torch.utils.data import DataLoader
from dataset import FaultGraphDataset
import torch.nn as nn
import torch
from torch import optim
import numpy as np
import dgl
from graph_classifier import NodeClassifier
from gnn import GNN
from ent_init import EntInit, EntInitAlarmFeat, MLP, EntInitOnlyName, EntInitWithAttr
import pickle as pkl


class Trainer(object):
    def __init__(self, args, train_fault_g, valid_fault_g, test_fault_g):
        self.args = args
        self.train_fault_g = train_fault_g
        self.valid_fault_g = valid_fault_g
        self.test_fault_g = test_fault_g
        map_dict = pkl.load(open(args.map_dict_path, 'rb'))
        self.alarm_name2idx = map_dict['alarm_name2idx']
        self.kpi_name2idx = map_dict['kpi_name2idx']

        self.name = args.task_name
        self.writer = SummaryWriter(os.path.join(args.tb_log_dir, self.name))
        self.logger = Log(args.log_dir, self.name).get_logger()
        self.logger.info(json.dumps({k: v for k, v in vars(args).items()}))

        # state dir
        self.state_path = os.path.join(args.state_dir, self.name)
        if not os.path.exists(self.state_path):
            os.makedirs(self.state_path)

        self.train_bs = args.train_bs
        self.eval_bs = args.eval_bs
        self.num_epoch = args.num_epoch
        self.lr = args.lr
        self.log_per_epoch = args.log_per_epoch
        self.valid_per_epoch = args.valid_per_epoch
        self.early_stop_patience = args.early_stop_patience

        self.train_loader = DataLoader(FaultGraphDataset(self.train_fault_g),
                                       batch_size=self.train_bs,
                                       shuffle=True,
                                       collate_fn=FaultGraphDataset.collate_fn,
                                       drop_last=False)

        self.valid_loader = DataLoader(FaultGraphDataset(self.valid_fault_g),
                                       batch_size=self.eval_bs,
                                       shuffle=False,
                                       collate_fn=FaultGraphDataset.collate_fn,
                                       drop_last=False)

        self.test_loader = DataLoader(FaultGraphDataset(self.test_fault_g),
                                       batch_size=self.eval_bs,
                                       shuffle=False,
                                       collate_fn=FaultGraphDataset.collate_fn,
                                       drop_last=False)

        # GNN
        # self.ent_init = EntInitAlarmFeat(args).to(self.args.gpu)
        if self.args.withAttr == 'True' :
            self.ent_init = EntInitWithAttr(args, map_dict).to(self.args.gpu)
        elif self.args.withAttr == 'False':
            self.ent_init = EntInit(args, map_dict).to(self.args.gpu)
        else:
            self.ent_init = EntInitOnlyName(args, map_dict).to(self.args.gpu)
        # self.ent_init = EntInit(args, map_dict).to(self.args.gpu)
        # self.gnn = GNN(args, len(self.alarm_name2idx) + len(self.kpi_name2idx), 1024, 512, nlayer=args.nlayer).to(self.args.gpu)
        # self.gnn = GNN(args, 200, 500, 200, nlayer=args.nlayer).to(self.args.gpu)
        if "_ent" in args.pt_emb_path:
            self.gnn = GNN(args, 256, 1024, 512, nlayer=args.nlayer).to(self.args.gpu)
        else: 
            self.gnn = GNN(args, 768, 1024, 512, nlayer=args.nlayer).to(self.args.gpu)
        # self.gnn = GNN(args, 10, 20, 10, nlayer=args.nlayer).to(self.args.gpu)
        # self.gnn = DGLGCN(args, node_dim=args.ent_dim, nlayer=args.nlayer).to(self.args.gpu)
        # self.graph_classifier = GraphClassifier(args).to(self.args.gpu)
        # self.graph_classifier = NodeClassifier(200, 50, 1).to(self.args.gpu)
        self.graph_classifier = NodeClassifier(512, 128, 1).to(self.args.gpu)
        # self.graph_classifier = MLP(10, 10, len(label_name2idx)).to(self.args.gpu)

        # optimizer
        self.optimizer = optim.Adam(
                list(self.ent_init.parameters()) +
                list(self.gnn.parameters()) +
                list(self.graph_classifier.parameters()),
                lr=self.lr)

    def loss_func(self, score, label):
        loss = torch.log(1 + torch.exp(-label*score))
        loss = torch.mean(loss)
        return loss

    def write_valid_acc(self, acc, e):
        self.writer.add_scalar("validation/acc", acc, e)

    def write_training_loss(self, loss, step):
        self.writer.add_scalar("training/loss", loss, step)

    def write_training_acc(self, acc, step):
        self.writer.add_scalar("training/acc", acc, step)

    def log_eval_res(self, res, info_str):
        self.logger.info(f"{info_str} | MR: {res['MR']:.4f}, H@1: {res['H@1']:.4f}, "
                         f"H@3: {res['H@3']:.4f}, H@5: {res['H@5']:.4f}")

    def log_train_loss_res(self, e, loss, res):
        self.logger.info(f"step: {e} | loss: {loss:.4f} | MR: {res['MR']:.4f}, "
                         f"H@1: {res['H@1']:.4f}, H@3: {res['H@3']:.4f}, H@5: {res['H@5']:.4f}")

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
        self.gnn.load_state_dict(state['gnn'])
        self.graph_classifier.load_state_dict(state['graph_classifier'])

    def get_ent_emb(self, g):
        ent_emb = self.ent_init(g)

        return ent_emb

    def get_score(self, ent_emb, rule_score):
        batch_ins_score = self.graph_classifier(ent_emb).reshape(-1)
        # score = batch_ins_score + self.graph_classifier.weight*rule_score
        score = batch_ins_score
        return score

    def get_rank(self, fault_g, score_list, label_list):
        rule_score = torch.tensor(fault_g.fault_pkg.rule_score).to(self.args.gpu)
        score = score_list * rule_score
        # score = rule_score
        # if self.args.use_rule == False:
        #     score = score_list
        sort_idx = torch.argsort(score, descending=True)
        rank = np.argwhere(label_list[sort_idx].numpy() == 1)[0].item() + 1

        return rank

    def split_emb(self, emb, split_list):
        split_list = [np.sum(split_list[0: i], dtype=np.int) for i in range(len(split_list) + 1)]
        emb_split = [emb[split_list[i]: split_list[i + 1]] for i in range(len(split_list) - 1)]
        return emb_split

    def train(self):
        best_epoch = 0
        bad_count = 0
        best_eval_res = {'MR': 100, 'H@1': 0, 'H@3': 0, 'H@5': 0}
        self.logger.info('start training')

        for e in range(1, self.num_epoch + 1):
            self.ent_init.train()
            self.gnn.train()
            self.graph_classifier.train()

            train_rank = []
            batch_losses = []
            for batch in self.train_loader:
                batch_g = dgl.batch([b.g for b in batch]).to(self.args.gpu)

                ent_emb = self.get_ent_emb(batch_g)

                ent_emb = self.gnn(batch_g, ent_emb)

                # loss
                batch_ins_score = self.graph_classifier(ent_emb).reshape(-1)
                batch_ins_label = torch.cat([torch.LongTensor(b.fault_ins_label) for b in batch]).to(self.args.gpu)

                batch_loss = self.loss_func(batch_ins_score, batch_ins_label)

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                batch_losses.append(batch_loss.item())

                # rank
                split_ins_score = self.split_emb(batch_ins_score, batch_g.batch_num_nodes().tolist())
                for batch_i, batch_data in enumerate(batch):
                    split_ins_label = batch_data.fault_ins_label
                    rank = self.get_rank(batch_data, split_ins_score[batch_i],
                                         torch.LongTensor(split_ins_label))
                    train_rank.append(rank)

            train_res = {'MR': np.mean(train_rank), 'H@1': np.sum(np.array(train_rank) == 1) / len(train_rank),
                         'H@3': np.sum(np.array(train_rank) <= 3) / len(train_rank),
                         'H@5': np.sum(np.array(train_rank) <= 5) / len(train_rank)}

            epoch_loss = sum(batch_losses) / len(batch_losses)

            self.write_training_loss(epoch_loss, e)

            if e % self.log_per_epoch == 0:
                self.log_train_loss_res(e, epoch_loss, train_res)

            if e % self.valid_per_epoch == 0 and e != 0:
                valid_res = self.eval()
                if valid_res['MR'] < best_eval_res['MR']:
                    best_eval_res = valid_res
                    best_epoch = e
                    self.logger.info(f"Best Model | MR: {valid_res['MR']:.4f}")
                    self.save_checkpoint(e, self.get_curr_state())
                    bad_count = 0
                else:
                    bad_count += 1
                    self.logger.info(f"Bad Count: {bad_count} | MR: {valid_res['MR']:.4f} | "
                                     f"Best Epoch: {best_epoch}, MR: {best_eval_res['MR']:.4f}")

            if bad_count >= self.early_stop_patience:
                self.logger.info('Early Stop at Epoch {}'.format(e))
                break

        self.logger.info('finish training')
        self.logger.info('save best model')
        self.save_model(best_epoch)

        self.log_eval_res(best_eval_res, 'Best Validation')
        self.before_test_load()
        test_res = self.eval(istest=True)

        return test_res

    def eval(self, istest=False):
        self.ent_init.eval()
        self.gnn.eval()
        self.graph_classifier.eval()

        if istest:
            data_loader = self.test_loader
            info_str = 'Testing'
        else:
            data_loader = self.valid_loader
            info_str = 'Validation'

        eval_rank = []
        for batch in data_loader:
            batch_g = dgl.batch([b.g for b in batch]).to(self.args.gpu)

            ent_emb = self.get_ent_emb(batch_g)
            ent_emb = self.gnn(batch_g, ent_emb)

            batch_ins_score = self.graph_classifier(ent_emb).reshape(-1)

            # rank
            split_ins_score = self.split_emb(batch_ins_score, batch_g.batch_num_nodes().tolist())
            for batch_i, batch_data in enumerate(batch):
                split_ins_label = batch_data.fault_ins_label
                rank = self.get_rank(batch_data, split_ins_score[batch_i],
                                     torch.LongTensor(split_ins_label))
                eval_rank.append(rank)

        eval_res = {'MR': np.mean(eval_rank), 'H@1': np.sum(np.array(eval_rank) == 1) / len(eval_rank),
                    'H@3': np.sum(np.array(eval_rank) <= 3) / len(eval_rank),
                    'H@5': np.sum(np.array(eval_rank) <= 5) / len(eval_rank)}

        self.log_eval_res(eval_res, info_str)

        return eval_res


