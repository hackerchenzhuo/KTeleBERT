import os
import os.path as osp
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
from easydict import EasyDict as edict
from tqdm import tqdm
import pdb
import pprint
import json
import pickle
from collections import defaultdict
import copy
from time import time

from config import cfg
from torchlight import initialize_exp, set_seed, get_dump_path
from src.data import load_data, load_data_kg, Collator_base, Collator_kg, SeqDataset, KGDataset, Collator_order, load_order_data
from src.utils import set_optim, Loss_log, add_special_token, time_trans
from src.distributed_utils import init_distributed_mode, dist_pdb, is_main_process, reduce_value, cleanup
import torch.distributed as dist

from itertools import cycle
from model import BertTokenizer, HWBert, KGEModel, OD_model, KE_model
import torch.multiprocessing
from torch.nn.parallel import DistributedDataParallel

# 默认用cuda就行


class Runner:
    def __init__(self, args, writer=None, logger=None, rank=0):
        self.datapath = edict()
        self.datapath.log_dir = get_dump_path(args)
        self.datapath.model_dir = os.path.join(self.datapath.log_dir, 'model')
        self.rank = rank
        # init code
        self.mlm_probability = args.mlm_probability
        self.args = args
        self.writer = writer
        self.logger = logger
        # 模型选择
        self.model_list = []
        self.model = HWBert(self.args)
        # 数据加载。添加special_token，同时把模型的embedding layer进行resize
        self.data_init()
        self.model.cuda()
        # 模型加载
        self.od_model, self.ke_model = None, None
        self.scaler = GradScaler()

        # 只要不是第一种训练策略就有新模型
        if self.args.train_strategy >= 2:
            self.ke_model = KE_model(self.args)
        if self.args.train_strategy >= 3:
            # TODO: 异常检测
            pass
        if self.args.train_strategy >= 4:
            self.od_model = OD_model(self.args)

        if self.args.model_name not in ['MacBert', 'TeleBert', 'TeleBert2', 'TeleBert3'] and not self.args.from_pretrain:
            # 如果不存在模型会直接返回None或者原始模型
            self.model = self._load_model(self.model, self.args.model_name)
            self.od_model = self._load_model(self.od_model, f"od_{self.args.model_name}")
            self.ke_model = self._load_model(self.ke_model, f"ke_{self.args.model_name}")
            # TODO: 异常检测

        # 测试的情况
        if self.args.only_test:
            self.dataloader_init(self.seq_test_set)
        else:
            # 训练
            if self.args.ernie_stratege > 0:
                self.args.mask_stratege = 'rand'
            # 初始化dataloader
            self.dataloader_init(self.seq_train_set, self.kg_train_set, self.order_train_set)
            if self.args.dist:
                # 并行训练需要权值共享
                self.model_sync()
            else:
                self.model_list = [model for model in [self.model, self.od_model, self.ke_model] if model is not None]

            self.optim_init(self.args)

    def model_sync(self):
        checkpoint_path = osp.join(self.args.data_path, "tmp", "initial_weights.pt")
        checkpoint_path_od = osp.join(self.args.data_path, "tmp", "initial_weights_od.pt")
        checkpoint_path_ke = osp.join(self.args.data_path, "tmp", "initial_weights_ke.pt")
        if self.rank == 0:
            torch.save(self.model.state_dict(), checkpoint_path)
            if self.od_model is not None:
                torch.save(self.od_model.state_dict(), checkpoint_path_od)
            if self.ke_model is not None:
                torch.save(self.ke_model.state_dict(), checkpoint_path_ke)
        dist.barrier()

        # if self.rank != 0:
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        self.model = self._model_sync(self.model, checkpoint_path)
        if self.od_model is not None:
            self.od_model = self._model_sync(self.od_model, checkpoint_path_od)
        if self.ke_model is not None:
            self.ke_model = self._model_sync(self.ke_model, checkpoint_path_ke)

    def _model_sync(self, model, checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.args.device))
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.args.device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.gpu], find_unused_parameters=True)
        self.model_list.append(model)
        model = model.module
        return model

    def optim_init(self, opt, total_step=None, accumulation_step=None):
        step_per_epoch = len(self.train_dataloader)
        # 占总step 10% 的warmup_steps
        opt.total_steps = int(step_per_epoch * opt.epoch) if total_step is None else int(total_step)
        opt.warmup_steps = int(opt.total_steps * 0.15)

        if self.rank == 0 and total_step is None:
            self.logger.info(f"warmup_steps: {opt.warmup_steps}")
            self.logger.info(f"total_steps: {opt.total_steps}")
            self.logger.info(f"weight_decay: {opt.weight_decay}")

        freeze_part = ['bert.encoder.layer.1.', 'bert.encoder.layer.2.', 'bert.encoder.layer.3.', 'bert.encoder.layer.4.'][:self.args.freeze_layer]
        self.optimizer, self.scheduler = set_optim(opt, self.model_list, freeze_part, accumulation_step)

    def data_init(self):
        # 载入数据, 两部分数据包括：载入mask loss部分的数据（序列化的数据） 和 载入triple loss部分的数据（三元组）
        # train_test_split: 训练集长度
        self.seq_train_set, self.seq_test_set, self.kg_train_set, self.kg_data = None, None, None, None
        self.order_train_set, self.order_test_set = None, None

        if self.args.train_strategy >= 1 and self.args.train_strategy <= 4:
            # 预训练 or multi task pretrain
            self.seq_train_set, self.seq_test_set, train_test_split = load_data(self.logger, self.args)
            if self.args.train_strategy >= 2:
                self.kg_train_set, self.kg_data = load_data_kg(self.logger, self.args)
            if self.args.train_strategy >= 3:
                # TODO: 异常检测的数据载入
                pass
            if self.args.train_strategy >= 4:
                self.order_train_set, self.order_test_set, train_test_split = load_order_data(self.logger, self.args)

        if self.args.dist and not self.args.only_test:
            # 测试不需要并行
            if self.args.train_strategy >= 1 and self.args.train_strategy <= 4:
                self.seq_train_sampler = torch.utils.data.distributed.DistributedSampler(self.seq_train_set)
                if self.args.train_strategy >= 2:
                    self.kg_train_sampler = torch.utils.data.distributed.DistributedSampler(self.kg_train_set)
                if self.args.train_strategy >= 3:
                    # TODO: 异常检测的数据载入
                    pass
                if self.args.train_strategy >= 4:
                    self.order_train_sampler = torch.utils.data.distributed.DistributedSampler(self.order_train_set)

            # self.seq_train_batch_sampler = torch.utils.data.BatchSampler(self.seq_train_sampler, self.args.batch_size, drop_last=True)
            # self.kg_train_batch_sampler = torch.utils.data.BatchSampler(self.kg_train_sampler, int(self.args.batch_size / 4), drop_last=True)

        # Tokenizer 载入
        model_name = self.args.model_name
        if self.args.model_name in ['TeleBert', 'TeleBert2', 'TeleBert3']:
            self.tokenizer = BertTokenizer.from_pretrained(osp.join(self.args.data_root, 'transformer', model_name), do_lower_case=True)
        else:
            if not osp.exists(osp.join(self.args.data_root, 'transformer', self.args.model_name)):
                model_name = 'MacBert'
            self.tokenizer = BertTokenizer.from_pretrained(osp.join(self.args.data_root, 'transformer', model_name), do_lower_case=True)

        # 添加special_token，同时把模型的embedding layer进行resize
        self.special_token = None
        # 单纯的telebert在测试时不需要特殊embedding
        if self.args.add_special_word and not (self.args.only_test and self.args.model_name in ['MacBert', 'TeleBert', 'TeleBert2', 'TeleBert3']):
            # tokenizer, special_token, norm_token
            # special_token 不应该被MASK
            self.tokenizer, special_token, _ = add_special_token(self.tokenizer, model=self.model.encoder, rank=self.rank, cache_path=self.args.specail_emb_path)
            # pdb.set_trace()
            self.special_token = [token.lower() for token in special_token]

    def _dataloader_dist(self, train_set, train_sampler, batch_size, collator):
        train_dataloader = DataLoader(
            train_set,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=self.args.workers,
            persistent_workers=True,
            drop_last=True,
            batch_size=batch_size,
            collate_fn=collator
        )
        return train_dataloader

    def _dataloader(self, train_set, batch_size, collator):
        train_dataloader = DataLoader(
            train_set,
            num_workers=self.args.workers,
            persistent_workers=True,
            shuffle=(self.args.only_test == 0),
            drop_last=(self.args.only_test == 0),
            batch_size=batch_size,
            collate_fn=collator
        )
        return train_dataloader

    def dataloader_init(self, train_set=None, kg_train_set=None, order_train_set=None):
        bs = self.args.batch_size
        bs_ke = self.args.batch_size_ke
        bs_od = self.args.batch_size_od
        bs_ad = self.args.batch_size_ad
        # 分布式
        if self.args.dist and not self.args.only_test:
            self.args.workers = min([os.cpu_count(), self.args.batch_size, self.args.workers])
            # if self.rank == 0:
            #     print(f'Using {self.args.workers} dataloader workers every process')

            if train_set is not None:
                seq_collator = Collator_base(self.args, tokenizer=self.tokenizer, special_token=self.special_token)
                self.train_dataloader = self._dataloader_dist(train_set, self.seq_train_sampler, bs, seq_collator)
            if kg_train_set is not None:
                kg_collator = Collator_kg(self.args, tokenizer=self.tokenizer, data=self.kg_data)
                self.train_dataloader_kg = self._dataloader_dist(kg_train_set, self.kg_train_sampler, bs_ke, kg_collator)
            if order_train_set is not None:
                order_collator = Collator_order(self.args, tokenizer=self.tokenizer)
                self.train_dataloader_order = self._dataloader_dist(order_train_set, self.order_train_sampler, bs_od, order_collator)
        else:
            if train_set is not None:
                seq_collator = Collator_base(self.args, tokenizer=self.tokenizer, special_token=self.special_token)
                self.train_dataloader = self._dataloader(train_set, bs, seq_collator)
            if kg_train_set is not None:
                kg_collator = Collator_kg(self.args, tokenizer=self.tokenizer, data=self.kg_data)
                self.train_dataloader_kg = self._dataloader(kg_train_set, bs_ke, kg_collator)
            if order_train_set is not None:
                order_collator = Collator_order(self.args, tokenizer=self.tokenizer)
                self.train_dataloader_order = self._dataloader(order_train_set, bs_od, order_collator)

    def dist_step(self, task=0):
        # 分布式训练需要额外step
        if self.args.dist:
            if task == 0:
                self.seq_train_sampler.set_epoch(self.dist_epoch)
            if task == 1:
                self.kg_train_sampler.set_epoch(self.dist_epoch)
            if task == 2:
                # TODO:异常检测
                pass
            if task == 3:
                self.order_train_sampler.set_epoch(self.dist_epoch)
            self.dist_epoch += 1

    def mask_rate_update(self, i):
        # 这种策略是曲线地增加 mask rate
        if self.args.mlm_probability_increase == "curve":
            self.args.mlm_probability += (i + 1) * ((self.args.final_mlm_probability - self.args.mlm_probability) / self.args.epoch)
        # 这种是线性的
        else:
            assert self.args.mlm_probability_increase == "linear"
            self.args.mlm_probability += (self.args.final_mlm_probability - self.mlm_probability) / self.args.epoch

        if self.rank == 0:
            self.logger.info(f"Moving Mlm_probability in next epoch to: {self.args.mlm_probability*100}%")

    def task_switch(self, training_strategy):
        # 同时训练或者策略1训练不需要切换任务，epoch也安装初始epoch就行
        if training_strategy == 1 or self.args.train_together:
            return (0, 0), None

        # 4 阶段
        # self.total_epoch -= 1

        for i in range(4):
            for task in range(training_strategy):
                if self.args.epoch_matrix[task][i] > 0:
                    self.args.epoch_matrix[task][i] -= 1
                    return (task, i), self.args.epoch_matrix[task][i] + 1

    def run(self):
        self.loss_log = Loss_log()
        self.curr_loss = 0.
        self.lr = self.args.lr
        self.curr_loss_dic = defaultdict(float)
        self.curr_kpi_loss_dic = defaultdict(float)
        self.loss_weight = [1, 1]
        self.kpi_loss_weight = [1, 1]
        self.step = 0
        # 不同task 的累计step
        self.total_step_sum = 0
        task_last = 0
        stage_last = 0
        self.dist_epoch = 0
        # 后面可以变成混合训练模式
        # self.total_epoch = self.args.epoch
        # --------- train -------------
        with tqdm(total=self.args.epoch) as _tqdm:  # 使用需要的参数对tqdm进行初始化
            for i in range(self.args.epoch):
                # 切换Task
                (task, stage), task_epoch = self.task_switch(self.args.train_strategy)
                self.dist_step(task)
                dataloader = self.task_dataloader_choose(task)
                # 并行
                if self.args.train_together and self.args.train_strategy > 1:
                    self.dataloader_list = ['#']
                    # 一个list 存下所有需要的dataloader的迭代
                    for t in range(1, self.args.train_strategy):
                        self.dist_step(t)
                        self.dataloader_list.append(iter(self.task_dataloader_choose(t)))

                if task != task_last or stage != stage_last:
                    self.step = 0
                    if self.rank == 0:
                        print(f"switch to task [{task}] in stage [{stage}]...")
                        if stage != stage_last:
                            # 每一个阶段结束保存一次
                            self._save_model(stage=f'_stg{stage_last}')
                    # task 转换状态时需要重新初始化优化器
                    # 并行训练或者单一task (0) 训练不需要切换opti
                    if task_epoch is not None:
                        self.optim_init(self.args, total_step=len(dataloader) * task_epoch, accumulation_step=self.args.accumulation_steps_dict[task])
                        task_last = task
                        stage_last = stage

                # 调整学习阶段
                if task == 0 and self.args.ernie_stratege > 0 and i >= self.args.ernie_stratege:
                    # 不会再触发第二次
                    self.args.ernie_stratege = 10000000
                    if self.rank == 0:
                        self.logger.info("switch to wwm stratege...")
                    self.args.mask_stratege = 'wwm'

                if self.args.mlm_probability != self.args.final_mlm_probability:
                    # 更新 MASK rate
                    # 初始化训练数据, 可以随epoch切换
                    # 混合训练
                    self.mask_rate_update(i)
                    self.dataloader_init(self.seq_train_set, self.kg_train_set, self.order_train_set)
                # -------------------------------
                # 针对task 进行训练
                self.train(_tqdm, dataloader, task)
                # -------------------------------
                _tqdm.update(1)

        # DONE: save or load
        if self.rank == 0:
            self.logger.info(f"min loss {self.loss_log.get_min_loss()}")
            # DONE: save or load
            if not self.args.only_test and self.args.save_model:
                self._save_model()

    def task_dataloader_choose(self, task):
        self.model.train()
        # 同时训练就用基础dataloader就行
        if task == 0:
            dataloader = self.train_dataloader
        elif task == 1:
            self.ke_model.train()
            dataloader = self.train_dataloader_kg
        elif task == 2:
            pass
        elif task == 3:
            self.od_model.train()
            dataloader = self.train_dataloader_order
        return dataloader
    # one time train

    def loss_output(self, batch, task):
        # -------- 模型输出 loss --------
        if task == 0:
            # 输出
            _output = self.model(batch)
            loss = _output['loss']
        elif task == 1:
            loss = self.ke_model(batch, self.model)
        elif task == 2:
            pass
        elif task == 3:
            # TODO: finetune的时候多任务 accumulation_steps 自适应
            # OD task
            emb = self.model.cls_embedding(batch[0], tp=self.args.plm_emb_type)
            loss, loss_dic = self.od_model(emb, batch[1].cuda())
            order_score = self.od_model.predict(emb)
            token_right = self.od_model.right_caculate(order_score, batch[1], threshold=0.5)
            self.loss_log.update_token(batch[1].shape[0], [token_right])
        return loss

    def train(self, _tqdm, dataloader, task=0):
        # cycle train
        loss_weight, kpi_loss_weight, kpi_loss_dict, _output = None, None, None, None
        # dataloader = zip(self.train_dataloader, cycle(self.train_dataloader_kg))
        self.loss_log.acc_init()
        # 如果self.train_dataloader比self.train_dataloader_kg长则会使得后者训练不完全
        accumulation_steps = self.args.accumulation_steps_dict[task]
        torch.cuda.empty_cache()

        for batch in dataloader:
            # with autocast():
            loss = self.args.mask_loss_scale * self.loss_output(batch, task)
            # 如果是同时训练的话使用迭代器的方法得到另外的epoch
            if self.args.train_together and self.args.train_strategy > 1:
                for t in range(1, self.args.train_strategy):
                    try:
                        batch = next(self.dataloader_list[t])
                    except StopIteration:
                        self.dist_step(t)
                        self.dataloader_list[t] = iter(self.task_dataloader_choose(t))
                        batch = next(self.dataloader_list[t])
                    # 选择对应的模型得到loss
                    # torch.cuda.empty_cache()
                    loss += self.loss_output(batch, t)
                    # torch.cuda.empty_cache()
            loss = loss / accumulation_steps
            self.scaler.scale(loss).backward()
            # loss.backward()
            if self.args.dist:
                loss = reduce_value(loss, average=True)
            # torch.cuda.empty_cache()
            self.step += 1
            self.total_step_sum += 1

            # -------- 模型统计 --------
            if not self.args.dist or is_main_process():
                self.output_statistic(loss, _output)
                acc_descrip = f"Acc: {self.loss_log.get_token_acc()}" if self.loss_log.get_token_acc() > 0 else ""
                _tqdm.set_description(f'Train | step [{self.step}/{self.args.total_steps}] {acc_descrip} LR [{self.lr}] Loss {self.loss_log.get_loss():.5f} ')
                if self.step % self.args.eval_step == 0 and self.step > 0:
                    self.loss_log.update(self.curr_loss)
                    self.update_loss_log()
            # -------- 梯度累计与模型更新 --------
            if self.step % accumulation_steps == 0 and self.step > 0:
                # 更新优化器
                self.scaler.unscale_(self.optimizer)
                for model in self.model_list:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)

                # self.optimizer.step()
                scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)

                self.scaler.update()
                skip_lr_sched = (scale > self.scaler.get_scale())
                if not skip_lr_sched:
                    # pdb.set_trace()
                    self.scheduler.step()

                if not self.args.dist or is_main_process():
                    # pdb.set_trace()
                    self.lr = self.scheduler.get_last_lr()[-1]
                    self.writer.add_scalars("lr", {"lr": self.lr}, self.total_step_sum)
                # 模型update
                for model in self.model_list:
                    model.zero_grad(set_to_none=True)

            if self.args.dist:
                torch.cuda.synchronize(self.args.device)
        return self.curr_loss, self.curr_loss_dic

    def output_statistic(self, loss, output):
        # 统计模型的各种输出
        self.curr_loss += loss.item()
        if output is None:
            return
        for key in output['loss_dic'].keys():
            self.curr_loss_dic[key] += output['loss_dic'][key]
        if 'kpi_loss_dict' in output and output['kpi_loss_dict'] is not None:
            for key in output['kpi_loss_dict'].keys():
                self.curr_kpi_loss_dic[key] += output['kpi_loss_dict'][key]
        if 'loss_weight' in output and output['loss_weight'] is not None:
            self.loss_weight = output['loss_weight']
        # 需要用dict来判断
        if 'kpi_loss_weight' in output and output['kpi_loss_weight'] is not None:
            self.kpi_loss_weight = output['kpi_loss_weight']

    def update_loss_log(self, task=0):
        # 把统计的模型各种输出存下来
        # https://zhuanlan.zhihu.com/p/382950853
        #  "mask_loss": self.curr_loss_dic['mask_loss'], "ke_loss": self.curr_loss_dic['ke_loss']
        vis_dict = {"train_loss": self.curr_loss}
        vis_dict.update(self.curr_loss_dic)
        self.writer.add_scalars("loss", vis_dict, self.total_step_sum)
        if self.loss_weight is not None:
            # 预训练
            loss_weight_dic = {}
            if self.args.train_strategy == 1:
                loss_weight_dic["mask"] = 1 / (self.loss_weight[0]**2)
                if self.args.use_NumEmb:
                    loss_weight_dic["kpi"] = 1 / (self.loss_weight[1]**2)
                    vis_kpi_dic = {"recover": 1 / (self.kpi_loss_weight[0]**2), "classifier": 1 / (self.kpi_loss_weight[1]**2)}
                    if self.args.contrastive_loss and len(self.kpi_loss_weight) > 2:
                        vis_kpi_dic.update({"contrastive": 1 / (self.kpi_loss_weight[2]**2)})
                    self.writer.add_scalars("kpi_loss_weight", vis_kpi_dic, self.total_step_sum)
                    self.writer.add_scalars("kpi_loss", self.curr_kpi_loss_dic, self.total_step_sum)
                self.writer.add_scalars("loss_weight", loss_weight_dic, self.total_step_sum)
            # TODO: Finetune

        # init log loss
        self.curr_loss = 0.
        for key in self.curr_loss_dic:
            self.curr_loss_dic[key] = 0.
        if len(self.curr_kpi_loss_dic) > 0:
            for key in self.curr_kpi_loss_dic:
                self.curr_kpi_loss_dic[key] = 0.

    # TODO: Finetune 阶段
    def eval(self):
        self.model.eval()
        torch.cuda.empty_cache()

    def mask_test(self, test_log):
        # 如果大于1 就无法mask测试
        assert self.args.train_ratio < 1
        topk = (1, 100, 500)
        test_log.acc_init(topk)
        # 做 mask 预测的时候需要进入训练模式，以获得随机mask的token
        self.args.only_test = 0
        self.dataloader_init(self.seq_test_set)
        # pdb.set_trace()
        sz_test = len(self.train_dataloader)
        loss_sum = 0
        with tqdm(total=sz_test) as _tqdm:  # 使用需要的参数对tqdm进行初始化
            for step, batch in enumerate(self.train_dataloader):
                # DONE: 写好mask_prediction实现mask预测
                with torch.no_grad():
                    token_num, token_right, loss = self.model.mask_prediction(batch, len(self.tokenizer), topk)
                test_log.update_token(token_num, token_right)
                loss_sum += loss
                # test_log.update_word(word_num, word_right)
                _tqdm.update(1)
                _tqdm.set_description(f'Test | step [{step}/{sz_test}] Top{topk} Token_Acc: {test_log.get_token_acc()}')
        print(f"perplexity: {loss_sum}")
        # 训练模式复位
        self.args.only_test = 1
        # if topk is not None:
        print(f"Top{topk} acc is {test_log.get_token_acc()}")

    def emb_generate(self, path_gen):
        assert len(self.args.path_gen) > 0 or path_gen is not None
        data_path = self.args.data_path
        if path_gen is None:
            path_gen = self.args.path_gen
        with open(osp.join(data_path, 'downstream_task', f'{path_gen}.json'), "r") as fp:
            data = json.load(fp)
        print(f"read file {path_gen} done!")
        test_set = SeqDataset(data)
        self.dataloader_init(test_set)
        sz_test = len(self.train_dataloader)
        all_emb_dic = defaultdict(list)
        emb_output = {}
        all_emb_ent = []
        # tps = ['cls', 'last_avg', 'last2avg', 'last3avg', 'first_last_avg']
        tps = ['cls', 'last_avg']
        # with tqdm(total=sz_test) as _tqdm:
        for step, batch in enumerate(self.train_dataloader):
            for tp in tps:
                with torch.no_grad():
                    batch_embedding = self.model.cls_embedding(batch, tp=tp)
                    # batch_embedding = self.model.cls_embedding(batch, tp=tp)
                    if tp in self.args.model_name and self.ke_model is not None:
                        batch_embedding_ent = self.ke_model.get_embedding(batch_embedding, is_ent=True)
                        # batch_embedding_ent = self.ke_model(batch, self.model)
                        batch_embedding_ent = batch_embedding_ent.cpu()
                        all_emb_ent.append(batch_embedding_ent)

                batch_embedding = batch_embedding.cpu()
                all_emb_dic[tp].append(batch_embedding)
            # _tqdm.update(1)
            # _tqdm.set_description(f'Test | step [{step}/{sz_test}]')
            torch.cuda.empty_cache()
        for tp in tps:
            emb_output[tp] = torch.cat(all_emb_dic[tp])
            assert emb_output[tp].shape[0] == len(data)
        if len(all_emb_ent) > 0:
            emb_output_ent = torch.cat(all_emb_ent)
        # 后缀
        save_path = osp.join(data_path, 'downstream_task', 'output')
        os.makedirs(save_path, exist_ok=True)
        for tp in tps:
            save_dir = osp.join(save_path, f'{path_gen}_emb_{self.args.model_name.replace("DistributedDataParallel", "")}_{tp}.pt')
            torch.save(emb_output[tp], save_dir)
        # 有训练好的实体embedding可使用
        if len(all_emb_ent) > 0:
            save_dir = osp.join(save_path, f'{path_gen}_emb_{self.args.model_name.replace("DistributedDataParallel", "")}_ent.pt')
            torch.save(emb_output_ent, save_dir)

    def KGE_test(self):
        # 直接用KG全集进行kge的测试
        sz_test = len(self.kg_train_set)
        # 先转换数据
        ent_set = set()
        rel_set = set()
        with tqdm(total=sz_test) as _tqdm:  # 使用需要的参数对tqdm进行初始化
            _tqdm.set_description('trans entity/relation ID')
            for batch in self.kg_train_set:
                ent_set.add(batch[0])
                ent_set.add(batch[2])
                rel_set.add(batch[1])
                _tqdm.update(1)
        all_ent, all_rel = list(ent_set), list(rel_set)
        nent, nrel = len(all_ent), len(all_rel)
        ent_dic, rel_dic = {}, {}
        for i in range(nent):
            ent_dic[all_ent[i]] = i
        for i in range(nrel):
            rel_dic[all_rel[i]] = i
        id_format_triple = []
        with tqdm(total=sz_test) as _tqdm:
            _tqdm.set_description('trans triple ID')
            for triple in self.kg_train_set:
                id_format_triple.append((ent_dic[triple[0]], rel_dic[triple[1]], ent_dic[triple[2]]))
                _tqdm.update(1)

        # pdb.set_trace()
        # 生成实体embedding并且保存
        ent_dataset = KGDataset(all_ent)
        rel_dataset = KGDataset(all_rel)

        ent_dataloader = DataLoader(
            ent_dataset,
            batch_size=self.args.batch_size * 32,
            num_workers=self.args.workers,
            persistent_workers=True,
            shuffle=False
        )
        rel_dataloader = DataLoader(
            rel_dataset,
            batch_size=self.args.batch_size * 32,
            num_workers=self.args.workers,
            persistent_workers=True,
            shuffle=False
        )

        sz_test = len(ent_dataloader) + len(rel_dataloader)
        with tqdm(total=sz_test) as _tqdm:
            ent_emb = []
            rel_emb = []
            step = 0
            _tqdm.set_description('get the ent embedding')
            with torch.no_grad():
                for batch in ent_dataloader:
                    batch = self.tokenizer.batch_encode_plus(
                        batch,
                        padding='max_length',
                        max_length=self.args.maxlength,
                        truncation=True,
                        return_tensors="pt",
                        return_token_type_ids=False,
                        return_attention_mask=True,
                        add_special_tokens=False
                    )

                    batch_emb = self.model.cls_embedding(batch, tp=self.args.plm_emb_type)
                    batch_emb = self.ke_model.get_embedding(batch_emb, is_ent=True)

                    ent_emb.append(batch_emb.cpu())
                    _tqdm.update(1)
                    step += 1
                    torch.cuda.empty_cache()
                    _tqdm.set_description(f'ENT emb:  [{step}/{sz_test}]')

                _tqdm.set_description('get the rel embedding')
                for batch in rel_dataloader:
                    batch = self.tokenizer.batch_encode_plus(
                        batch,
                        padding='max_length',
                        max_length=self.args.maxlength,
                        truncation=True,
                        return_tensors="pt",
                        return_token_type_ids=False,
                        return_attention_mask=True,
                        add_special_tokens=False
                    )
                    batch_emb = self.model.cls_embedding(batch, tp=self.args.plm_emb_type)
                    batch_emb = self.ke_model.get_embedding(batch_emb, is_ent=False)
                    # batch_emb = self.model.get_embedding(batch, is_ent=False)
                    rel_emb.append(batch_emb.cpu())
                    _tqdm.update(1)
                    step += 1
                    torch.cuda.empty_cache()
                    _tqdm.set_description(f'REL emb: [{step}/{sz_test}]')

        all_ent_emb = torch.cat(ent_emb).cuda()
        all_rel_emb = torch.cat(rel_emb).cuda()
        # embedding：emb_output
        # dim 256
        kge_model_for_test = KGEModel(nentity=len(all_ent), nrelation=len(all_rel), hidden_dim=self.args.ke_dim,
                                      gamma=self.args.ke_margin, entity_embedding=all_ent_emb, relation_embedding=all_rel_emb).cuda()
        if self.args.ke_test_num > 0:
            test_triples = id_format_triple[:self.args.ke_test_num]
        else:
            test_triples = id_format_triple
        with torch.no_grad():
            metrics = kge_model_for_test.test_step(test_triples=test_triples, all_true_triples=id_format_triple, args=self.args, nentity=len(all_ent), nrelation=len(all_rel))
        # pdb.set_trace()
        print(f"result:{metrics}")

    def OD_test(self):
        # data_path = self.args.data_path
        # with open(osp.join(data_path, f'{self.args.order_test_name}.json'), "r") as fp:
        #     data = json.load(fp)
        self.od_model.eval()
        test_log = Loss_log()
        test_log.acc_init()
        sz_test = len(self.train_dataloader)
        all_emb_ent = []
        with tqdm(total=sz_test) as _tqdm:  # 使用需要的参数对tqdm进行初始化
            for step, batch in enumerate(self.train_dataloader):
                with torch.no_grad():
                    emb = self.model.cls_embedding(batch[0], tp=self.args.plm_emb_type)
                    out_emb = self.od_model.encode(emb)
                    emb_cpu = out_emb.cpu()
                    all_emb_ent.append(emb_cpu)
                    order_score = self.od_model.predict(emb)
                    token_right = self.od_model.right_caculate(order_score, batch[1], threshold=self.args.order_threshold)
                test_log.update_token(batch[1].shape[0], [token_right])
                _tqdm.update(1)
                _tqdm.set_description(f'Test | step [{step}/{sz_test}] Acc: {test_log.get_token_acc()}')

        emb_output = torch.cat(all_emb_ent)
        data_path = self.args.data_path
        save_path = osp.join(data_path, 'downstream_task', 'output')
        os.makedirs(save_path, exist_ok=True)
        save_dir = osp.join(save_path, f'ratio{self.args.train_ratio}_{emb_output.shape[0]}emb_{self.args.model_name.replace("DistributedDataParallel", "")}.pt')
        torch.save(emb_output, save_dir)
        print(f"save {emb_output.shape[0]} embeddings done...")

    @ torch.no_grad()
    def test(self, path_gen=None):
        test_log = Loss_log()
        self.model.eval()
        if not (self.args.mask_test or self.args.embed_gen or self.args.ke_test or len(self.args.order_test_name) > 0):
            return
        if self.args.mask_test:
            self.mask_test(test_log)
        if self.args.embed_gen:
            self.emb_generate(path_gen)
        if self.args.ke_test:
            self.KGE_test()
        if len(self.args.order_test_name) > 0:
            runner.OD_test()

    def _load_model(self, model, name):
        if model is None:
            return None
        # 没有训练过
        _name = name if name[:3] not in ['od_', 'ke_'] else name[3:]
        save_path = osp.join(self.args.data_path, 'save', _name)
        save_name = osp.join(save_path, f'{name}.pkl')
        if not osp.exists(save_path) or not osp.exists(save_name):
            return model.cuda()
        # 载入模型
        if 'Distribute' in self.args.model_name:
            model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(os.path.join(save_name), map_location=self.args.device).items()})
        else:
            model.load_state_dict(torch.load(save_name, map_location=self.args.device))
        model.cuda()
        if self.rank == 0:
            print(f"loading model [{name}.pkl] done!")

        return model

    def _save_model(self, stage=''):
        model_name = type(self.model).__name__
        # TODO: path
        save_path = osp.join(self.args.data_path, 'save')
        os.makedirs(save_path, exist_ok=True)
        if self.args.train_strategy == 1:
            save_name = f'{self.args.exp_name}_{self.args.exp_id}_s{self.args.random_seed}{stage}'
        else:
            save_name = f'{self.args.exp_name}_{self.args.exp_id}_s{self.args.random_seed}_{self.args.plm_emb_type}{stage}'
        save_path = osp.join(save_path, save_name)
        os.makedirs(save_path, exist_ok=True)
        # 预训练模型保存
        self._save(self.model, save_path, save_name)

        # 下游模型保存
        save_name_od = f'od_{save_name}'
        self._save(self.od_model, save_path, save_name_od)
        save_name_ke = f'ke_{save_name}'
        self._save(self.ke_model, save_path, save_name_ke)
        return save_path

    def _save(self, model, save_path, save_name):
        if model is None:
            return
        if self.args.save_model:
            torch.save(model.state_dict(), osp.join(save_path, f'{save_name}.pkl'))
            print(f"saving {save_name} done!")

        if self.args.save_pretrain and not save_name.startswith('od_') and not save_name.startswith('ke_'):
            self.tokenizer.save_pretrained(osp.join(self.args.plm_path, f'{save_name}'))
            self.model.encoder.save_pretrained(osp.join(self.args.plm_path, f'{save_name}'))
            print(f"saving [pretrained] {save_name} done!")


if __name__ == '__main__':
    cfg = cfg()
    cfg.get_args()
    cfgs = cfg.update_train_configs()
    set_seed(cfgs.random_seed)
    # 初始化各进程环境
    # pdb.set_trace()
    if cfgs.dist and not cfgs.only_test:
        init_distributed_mode(args=cfgs)
        # cfgs.lr *= cfgs.world_size
        # cfgs.ke_lr *= cfgs.world_size
    else:
        # 下面这条语句在并行的时候可能内存泄漏，导致无法停止
        torch.multiprocessing.set_sharing_strategy('file_system')
    rank = cfgs.rank

    writer, logger = None, None
    if rank == 0:
        # 如果并行则只有一种情况打印
        logger = initialize_exp(cfgs)
        logger_path = get_dump_path(cfgs)
        cfgs.time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        comment = f'bath_size={cfgs.batch_size} exp_id={cfgs.exp_id}'
        if not cfgs.no_tensorboard and not cfgs.only_test:
            writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard', cfgs.time_stamp), comment=comment)

    cfgs.device = torch.device(cfgs.device)

    # -----  Begin ----------
    runner = Runner(cfgs, writer, logger, rank)

    if cfgs.only_test:
        if cfgs.embed_gen:
            # 不需要生成的先搞定
            if cfgs.mask_test or cfgs.ke_test:
                runner.args.embed_gen = 0
                runner.test()
                runner.args.embed_gen = 1
            # gen_dir = ['yht_data_merge', 'yht_data_whole5gc', 'yz_data_whole5gc', 'yz_data_merge', 'zyc_data_merge', 'zyc_data_whole5gc']
            gen_dir = ['yht_serialize_withAttribute', 'yht_serialize_withoutAttr', 'yht_name_serialize', 'zyc_serialize_withAttribute', 'zyc_serialize_withoutAttr', 'zyc_name_serialize',
                       'yz_serialize_withAttribute', 'yz_serialize_withoutAttr', 'yz_name_serialize', 'yz_serialize_net']
            # gen_dir = ['zyc_serialize_withAttribute', 'zyc_normal_serialize', 'zyc_data_whole5gc', 'zyc_data_merge', 'yht_normal_serialize',
            #            'yht_serialize_withAttribute', 'yz_serialize_withAttribute', 'yz_serialize_net', 'yz_normal_serialize']
            runner.args.mask_test, runner.args.ke_test = 0, 0
            for item in gen_dir:
                runner.test(item)
        else:
            runner.test()
    else:
        runner.run()

    # -----  End ----------
    if not cfgs.no_tensorboard and not cfgs.only_test and rank == 0:
        writer.close()
        logger.info("done!")

    if cfgs.dist and not cfgs.only_test:
        dist.barrier()
        dist.destroy_process_group()
        # print("shut down...")
