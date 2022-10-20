import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import pdb
from torch.utils.data import DataLoader
from collections import defaultdict
import os.path as osp
import json


class KE_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        """
        triple task: mask tail entity, total entity size-class classification 
        """
        """
        :param hidden: BERT model output size
        """
        self.args = args
        self.ke_dim = args.ke_dim

        self.linear_ent = nn.Linear(args.hidden_size, self.ke_dim)
        self.linear_rel = nn.Linear(args.hidden_size, self.ke_dim)

        self.ke_margin = nn.Parameter(
            torch.Tensor([args.ke_margin]),
            requires_grad=False
        )

    def forward(self, batch, hw_model):
        batch_triple = batch
        pos_sample = batch_triple["positive_sample"]
        neg_sample = batch_triple["negative_sample"]
        neg_index = batch_triple["neg_index"]
        
        # 节省显存
        all_entity = []
        all_entity_mask = []
        for i in range(3):
            all_entity.append(pos_sample[i]['input_ids'])
            all_entity_mask.append(pos_sample[i]['attention_mask'])
        
        all_entity = torch.cat(all_entity)
        all_entity_mask = torch.cat(all_entity_mask)
        entity_data = {'input_ids':all_entity, 'attention_mask':all_entity_mask}
        entity_emb = hw_model.cls_embedding(entity_data, tp=self.args.plm_emb_type)

        bs = pos_sample[0]['input_ids'].shape[0]
        pos_sample_emb= [entity_emb[:bs], entity_emb[bs:2*bs], entity_emb[2*bs:3*bs]]
        neg_sample_emb = entity_emb[neg_index]
        mode = batch_triple["mode"]
        # pos_score = self.get_score(pos_sample, hw_model)
        # neg_score = self.get_score(pos_sample, hw_model, neg_sample, mode)
        pos_score = self.get_score(pos_sample_emb, hw_model)
        neg_score = self.get_score(pos_sample_emb, hw_model, neg_sample_emb, mode)
        triple_loss = self.adv_loss(pos_score, neg_score, self.args)

        return triple_loss

        # pdb.set_trace()
        # return emb.div_(emb.detach().norm(p=1, dim=-1, keepdim=True))

# KE loss
    def tri2emb(self, triples, hw_model, negs=None, mode="single"):
        """Get embedding of triples.
        This function get the embeddings of head, relation, and tail
        respectively. each embedding has three dimensions.
        Args:
            triples (tensor): This tensor save triples id, which dimension is 
                [triples number, 3].
            negs (tensor, optional): This tenosr store the id of the entity to 
                be replaced, which has one dimension. when negs is None, it is 
                in the test/eval phase. Defaults to None.
            mode (str, optional): This arg indicates that the negative entity 
                will replace the head or tail entity. when it is 'single', it 
                means that entity will not be replaced. Defaults to 'single'.
        Returns:
            head_emb: Head entity embedding.
            relation_emb: Relation embedding.
            tail_emb: Tail entity embedding.
        """

        if mode == "single":
            head_emb = self.get_embedding(triples[0]).unsqueeze(1)  # [bs, 1, dim]
            relation_emb = self.get_embedding(triples[1], is_ent=False).unsqueeze(1)  # [bs, 1, dim]
            tail_emb = self.get_embedding(triples[2]).unsqueeze(1)  # [bs, 1, dim]

        elif mode == "head-batch" or mode == "head_predict":
            if negs is None:  # 说明这个时候是在evluation，所以需要直接用所有的entity embedding
                # TODO：暂时不考虑KGC的测试情况
                head_emb = self.ent_emb.weight.data.unsqueeze(0)  # [1, num_ent, dim]
            else:
                head_emb = self.get_embedding(negs).reshape(-1, self.args.neg_num, self.args.ke_dim)  # [bs, num_neg, dim]
            relation_emb = self.get_embedding(triples[1], is_ent=False).unsqueeze(1)  # [bs, 1, dim]
            tail_emb = self.get_embedding(triples[2]).unsqueeze(1)  # [bs, 1, dim]

        elif mode == "tail-batch" or mode == "tail_predict":
            head_emb = self.get_embedding(triples[0]).unsqueeze(1)  # [bs, 1, dim]
            relation_emb = self.get_embedding(triples[1], is_ent=False).unsqueeze(1)  # [bs, 1, dim]
            if negs is None:
                tail_emb = self.ent_emb.weight.data.unsqueeze(0)  # [1, num_ent, dim]
            else:
                # pdb.set_trace()
                tail_emb = self.get_embedding(negs).reshape(-1, self.args.neg_num, self.args.ke_dim)  # [bs, num_neg, dim]

        return head_emb, relation_emb, tail_emb

    def get_embedding(self, inputs, is_ent=True):
        # pdb.set_trace()
        if is_ent:
            return self.linear_ent(inputs)
        else:
            return self.linear_rel(inputs)

    def score_func(self, head_emb, relation_emb, tail_emb):
        """Calculating the score of triples.

        The formula for calculating the score is :math:`\gamma - ||h + r - t||_F`
        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
            mode: Choose head-predict or tail-predict.
        Returns:
            score: The score of triples.
        """
        score = (head_emb + relation_emb) - tail_emb
        # pdb.set_trace()
        score = self.ke_margin.item() - torch.norm(score, p=1, dim=-1)
        return score

    def get_score(self, triples, hw_model, negs=None, mode='single'):
        """The functions used in the training phase

        Args:
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].
            negs: Negative samples, defaults to None.
            mode: Choose head-predict or tail-predict, Defaults to 'single'.

        Returns:
            score: The score of triples.
        """
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, hw_model, negs, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb)

        return score

    def adv_loss(self, pos_score, neg_score, args):
        """Negative sampling loss with self-adversarial training. In math:

        L=-\log \sigma\left(\gamma-d_{r}(\mathbf{h}, \mathbf{t})\right)-\sum_{i=1}^{n} p\left(h_{i}^{\prime}, r, t_{i}^{\prime}\right) \log \sigma\left(d_{r}\left(\mathbf{h}_{i}^{\prime}, \mathbf{t}_{i}^{\prime}\right)-\gamma\right)

        Args:
            pos_score: The score of positive samples.
            neg_score: The score of negative samples.
            subsampling_weight: The weight for correcting pos_score and neg_score.

        Returns:
            loss: The training loss for back propagation.
        """
        neg_score = (F.softmax(neg_score * args.adv_temp, dim=1).detach()
                     * F.logsigmoid(-neg_score)).sum(dim=1)  # shape:[bs]
        pos_score = F.logsigmoid(pos_score).view(neg_score.shape[0])  # shape:[bs]
        positive_sample_loss = - pos_score.mean()
        negative_sample_loss = - neg_score.mean()
        loss = (positive_sample_loss + negative_sample_loss) / 2
        return loss


class KGEModel(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma, entity_embedding, relation_embedding):
        super(KGEModel, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding

        assert self.relation_embedding.shape[0] == nrelation
        assert self.entity_embedding.shape[0] == nentity

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        score = self.TransE(head, relation, tail, mode)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=-1)
        return score

    @torch.no_grad()
    def test_step(self, test_triples, all_true_triples, args, nentity, nrelation):
        '''
        Evaluate the model on test or valid datasets
        '''
        # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        # Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            KGTestDataset(
                test_triples,
                all_true_triples,
                nentity,
                nrelation,
                'head-batch'
            ),
            batch_size=args.batch_size,
            num_workers=args.workers,
            persistent_workers=True,
            collate_fn=KGTestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            KGTestDataset(
                test_triples,
                all_true_triples,
                nentity,
                nrelation,
                'tail-batch'
            ),
            batch_size=args.batch_size,
            num_workers=args.workers,
            persistent_workers=True,
            collate_fn=KGTestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        # pdb.set_trace()
        with tqdm(total=total_steps) as _tqdm:
            _tqdm.set_description(f'eval KGC')
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:

                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    score = self.forward((positive_sample, negative_sample), mode)
                    score += filter_bias

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        # ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero(as_tuple=False)
                        assert ranking.size(0) == 1

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    # if step % args.test_log_steps == 0:
                    #     logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                    _tqdm.update(1)
                    _tqdm.set_description(f'KGC Eval:')
                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics


# 专门为KGE的测试设计一个dataset
class KGTestDataset(torch.utils.data.Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode, head4rel_tail=None, tail4head_rel=None):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples

        # 需要统计得到
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

        # 给定关系尾实体对应头实体
        # print("build head4rel_tail")
        # self.head4rel_tail = self.find_head4rel_tail()
        # print("build tail4head_rel")
        # self.tail4head_rel = self.find_tail4head_rel()

    def __len__(self):
        return self.len

    def find_head4rel_tail(self):
        ans = defaultdict(list)
        for (h, r, t) in self.triple_set:
            ans[(r, t)].append(h)
        return ans

    def find_tail4head_rel(self):
        ans = defaultdict(list)
        for (h, r, t) in self.triple_set:
            ans[(h, r)].append(t)
        return ans

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-100, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-100, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
        # if self.mode == 'head-batch':
            #
        #     tmp = [(0, rand_head) if rand_head not in  self.head4rel_tail[(relation, tail)]
        #            else (-100, head) for rand_head in range(self.nentity)]
        #     tmp[head] = (0, head)
        # elif self.mode == 'tail-batch':
        #     tmp = [(0, rand_tail) if rand_tail not in self.tail4head_rel[(head, relation)]
        #            else (-100, tail) for rand_tail in range(self.nentity)]
        #     tmp[tail] = (0, tail)
        # else:
        #     raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
