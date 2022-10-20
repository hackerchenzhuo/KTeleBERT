import torch
import random
import json
import numpy as np
import pdb
import os.path as osp
from model import BertTokenizer
import torch.distributed as dist


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data, chi_ref=None, kpi_ref=None):
        self.data = data
        self.chi_ref = chi_ref
        self.kpi_ref = kpi_ref

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.chi_ref is not None:
            chi_ref = self.chi_ref[index]
        else:
            chi_ref = None

        if self.kpi_ref is not None:
            kpi_ref = self.kpi_ref[index]
        else:
            kpi_ref = None

        return sample, chi_ref, kpi_ref


class OrderDataset(torch.utils.data.Dataset):
    def __init__(self, data, kpi_ref=None):
        self.data = data
        self.kpi_ref = kpi_ref

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.kpi_ref is not None:
            kpi_ref = self.kpi_ref[index]
        else:
            kpi_ref = None

        return sample, kpi_ref


class KGDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        sample = self.data[index]
        return sample

# TODO: 重构 DataCollatorForLanguageModeling


class Collator_base(object):
    # TODO: 定义 collator，模仿Lako
    # 完成mask，padding
    def __init__(self, args, tokenizer, special_token=None):
        self.tokenizer = tokenizer
        if special_token is None:
            self.special_token = ['[SEP]', '[MASK]', '[ALM]', '[KPI]', '[CLS]', '[LOC]', '[EOS]', '[ENT]', '[ATTR]', '[NUM]', '[REL]', '|', '[DOC]']
        else:
            self.special_token = special_token

        self.text_maxlength = args.maxlength
        self.mlm_probability = args.mlm_probability
        self.args = args
        if self.args.special_token_mask:
            self.special_token = ['|', '[NUM]']

        if not self.args.only_test and self.args.use_mlm_task:
            if args.mask_stratege == 'rand':
                self.mask_func = self.torch_mask_tokens
            else:
                if args.mask_stratege == 'wwm':
                    # 必须使用special_word, 因为这里的wwm基于分词
                    if args.rank == 0:
                        print("use word-level Mask ...")
                    assert args.add_special_word == 1
                    self.mask_func = self.wwm_mask_tokens
                else:  # domain
                    if args.rank == 0:
                        print("use token-level Mask ...")
                    self.mask_func = self.domain_mask_tokens

    def __call__(self, batch):
        # 把 batch 中的数值提取出，用specail token 替换
        # 把数值信息，以及数值的位置信息单独通过list传进去
        # 后面训练的阶段直接把数值插入embedding的位置
        # 数值不参与 mask
        # wwm的时候可以把chinese ref 随batch一起输入
        kpi_ref = None
        if self.args.use_NumEmb:
            kpi_ref = [item[2] for item in batch]
        # if self.args.mask_stratege != 'rand':
        chinese_ref = [item[1] for item in batch]
        batch = [item[0] for item in batch]
        # 此时batch不止有字符串
        batch = self.tokenizer.batch_encode_plus(
            batch,
            padding='max_length',
            max_length=self.text_maxlength,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
            add_special_tokens=False
        )
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        # self.torch_mask_tokens

        # if batch["input_ids"].shape[1] != 128:
        #     pdb.set_trace()
        if chinese_ref is not None:
            batch["chinese_ref"] = chinese_ref
        if kpi_ref is not None:
            batch["kpi_ref"] = kpi_ref

        # 训练需要 mask

        if not self.args.only_test and self.args.use_mlm_task:
            batch["input_ids"], batch["labels"] = self.mask_func(
                batch, special_tokens_mask=special_tokens_mask
            )
        else:
            # 非训练状态
            # 且不用MLM进行训练
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        return batch

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        if "input_ids" in inputs:
            inputs = inputs["input_ids"]
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        # pdb.set_trace()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def wwm_mask_tokens(self, inputs, special_tokens_mask=None):
        mask_labels = []
        ref_tokens = inputs["chinese_ref"]
        input_ids = inputs["input_ids"]
        sz = len(input_ids)

        # 把input id 先恢复到token
        for i in range(sz):
            # 这里的主体是读入的ref，但是可能存在max_len不统一的情况
            mask_labels.append(self._whole_word_mask(ref_tokens[i]))

        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, self.text_maxlength, pad_to_multiple_of=None)
        inputs, labels = self.torch_mask_tokens_4wwm(input_ids, batch_mask)
        return inputs, labels

    # input_tokens: List[str]
    def _whole_word_mask(self, input_tokens, max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        assert isinstance(self.tokenizer, (BertTokenizer))
        # 输入是 [..., ..., ..., ...] 格式
        cand_indexes = []
        cand_token = []

        for i, token in enumerate(input_tokens):
            if i >= self.text_maxlength - 1:
                # 不能超过最大值，截断一下
                break
            if token.lower() in self.special_token:
                # special token 的词不应该被mask
                continue
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
                cand_token.append(i)
            else:
                cand_indexes.append([i])
                cand_token.append(i)

        random.shuffle(cand_indexes)
        # 原来是：input_tokens
        # 但是这里的特殊token很多，因此提前去掉了特殊token
        # 这里的15%是去掉了特殊token的15%。+2的原因是把CLS SEP两个 flag的长度加上
        num_to_predict = min(max_predictions, max(1, int(round((len(cand_token) + 2) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            # 到达长度了结束
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            # 不能让其长度大于15%，最多等于
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                # 不考虑重叠的token进行mask
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            # 一般不会出现，因为过程中避免重复了
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
            # 不能超过最大值，截断
        mask_labels = [1 if i in covered_indexes else 0 for i in range(min(len(input_tokens), self.text_maxlength))]

        return mask_labels

        # 确定这里面需要mask的：置0/1

        # 调用 self.torch_mask_tokens

        #
        pass

    def torch_mask_tokens_4wwm(self, inputs, mask_labels):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        # if "input_ids" in inputs:
        #     inputs = inputs["input_ids"]
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]

        if len(special_tokens_mask[0]) != probability_matrix.shape[1]:
            print(f"len(special_tokens_mask[0]): {len(special_tokens_mask[0])}")
            print(f"probability_matrix.shape[1]): {probability_matrix.shape[1]}")
            print(f'max len {self.text_maxlength}')
            print(f"pad_token_id: {self.tokenizer.pad_token_id}")
            # if self.args.rank != in_rank:
            if self.args.dist:
                dist.barrier()
                pdb.set_trace()
            else:
                pdb.set_trace()

        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 这里的wwm，每次 mask/替换/不变的时候单位不是一体的，会拆开
        # 其实不太合理，但是也没办法

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    # TODO: 按区域cell 进行mask

    def domain_mask_tokens(self, inputs, special_tokens_mask=None):
        pass


class Collator_kg(object):
    # TODO: 定义 collator，模仿Lako
    # 完成 随机减少一部分属性
    def __init__(self, args, tokenizer, data):
        self.tokenizer = tokenizer
        self.text_maxlength = args.maxlength
        self.cross_sampling_flag = 0
        # ke 的bs 是正常bs的四分之一
        self.neg_num = args.neg_num
        # 负样本不能在全集中
        self.data = data
        self.args = args

    def __call__(self, batch):
        # 先编码成可token形式避免重复编码
        outputs = self.sampling(batch)

        return outputs

    def sampling(self, data):
        """Filtering out positive samples and selecting some samples randomly as negative samples.

        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """
        batch_data = {}
        neg_ent_sample = []

        self.cross_sampling_flag = 1 - self.cross_sampling_flag

        head_list = []
        rel_list = []
        tail_list = []
        # pdb.set_trace()
        if self.cross_sampling_flag == 0:
            batch_data['mode'] = "head-batch"
            for index, (head, relation, tail) in enumerate(data):
                # in batch negative
                neg_head = self.find_neghead(data, index, relation, tail)
                neg_ent_sample.extend(random.sample(neg_head, self.neg_num))
                head_list.append(head)
                rel_list.append(relation)
                tail_list.append(tail)
        else:
            batch_data['mode'] = "tail-batch"
            for index, (head, relation, tail) in enumerate(data):
                neg_tail = self.find_negtail(data, index, relation, head)
                neg_ent_sample.extend(random.sample(neg_tail, self.neg_num))

                head_list.append(head)
                rel_list.append(relation)
                tail_list.append(tail)

        neg_ent_batch = self.batch_tokenizer(neg_ent_sample)
        head_batch = self.batch_tokenizer(head_list)
        rel_batch = self.batch_tokenizer(rel_list)
        tail_batch = self.batch_tokenizer(tail_list)

        ent_list = head_list + rel_list + tail_list
        ent_dict = {k: v for v, k in enumerate(ent_list)}
        # 用来索引负样本
        neg_index = torch.tensor([ent_dict[i] for i in neg_ent_sample])
        # pos_head_index = torch.tensor(list(range(len(head_list)))

        batch_data["positive_sample"] = (head_batch, rel_batch, tail_batch)
        batch_data['negative_sample'] = neg_ent_batch
        batch_data['neg_index'] = neg_index
        return batch_data

    def batch_tokenizer(self, input_list):
        return self.tokenizer.batch_encode_plus(
            input_list,
            padding='max_length',
            max_length=self.text_maxlength,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
            add_special_tokens=False
        )

    def find_neghead(self, data, index, rel, ta):
        head_list = []
        for i, (head, relation, tail) in enumerate(data):
            # 负样本不能被包含
            if i != index and [head, rel, ta] not in self.data:
                head_list.append(head)
        # 可能存在负样本不够的情况
        # 自补齐
        while len(head_list) < self.neg_num:
            head_list.extend(random.sample(head_list, min(self.neg_num - len(head_list), len(head_list))))

        return head_list

    def find_negtail(self, data, index, rel, he):
        tail_list = []
        for i, (head, relation, tail) in enumerate(data):
            if i != index and [he, rel, tail] not in self.data:
                tail_list.append(tail)
        # 可能存在负样本不够的情况
        # 自补齐
        while len(tail_list) < self.neg_num:
            tail_list.extend(random.sample(tail_list, min(self.neg_num - len(tail_list), len(tail_list))))
        return tail_list

# 载入mask loss部分的数据


def load_data(logger, args):

    data_path = args.data_path

    data_name = args.seq_data_name
    with open(osp.join(data_path, f'{data_name}_cws.json'), "r") as fp:
        data = json.load(fp)
    if args.rank == 0:
        logger.info(f"[Start] Loading Seq dataset: [{len(data)}]...")
    random.shuffle(data)

    # data = data[:10000]
    # pdb.set_trace()
    train_test_split = int(args.train_ratio * len(data))
    # random.shuffle(x)
    # 训练/测试期间不应该打乱
    train_data = data[0: train_test_split]
    test_data = data[train_test_split: len(data)]

    # 测试的时候也可能用到其实 not args.only_test
    if args.use_mlm_task:
        # if args.mask_stratege != 'rand':
        # 读领域词汇
        if args.rank == 0:
            print("using the domain words .....")
        domain_file_path = osp.join(args.data_path, f'{data_name}_chinese_ref.json')
        with open(domain_file_path, 'r') as f:
            chinese_ref = json.load(f)
    # train_test_split=len(data)
        chi_ref_train = chinese_ref[:train_test_split]
        chi_ref_eval = chinese_ref[train_test_split:]
    else:
        chi_ref_train = None
        chi_ref_eval = None

    if args.use_NumEmb:
        if args.rank == 0:
            print("using the kpi and num  .....")

        kpi_file_path = osp.join(args.data_path, f'{data_name}_kpi_ref.json')
        with open(kpi_file_path, 'r') as f:
            kpi_ref = json.load(f)
        kpi_ref_train = kpi_ref[:train_test_split]
        kpi_ref_eval = kpi_ref[train_test_split:]
    else:
        # num_ref_train = None
        # num_ref_eval = None
        kpi_ref_train = None
        kpi_ref_eval = None

    # pdb.set_trace()
    test_set = None
    train_set = SeqDataset(train_data, chi_ref=chi_ref_train, kpi_ref=kpi_ref_train)
    if len(test_data) > 0:
        test_set = SeqDataset(test_data, chi_ref=chi_ref_eval, kpi_ref=kpi_ref_eval)
    if args.rank == 0:
        logger.info("[End] Loading Seq dataset...")
    return train_set, test_set, train_test_split

# 载入triple loss部分的数据


def load_data_kg(logger, args):
    data_path = args.data_path
    if args.rank == 0:
        logger.info("[Start] Loading KG dataset...")
    # # 三元组
    # with open(osp.join(data_path, '5GC_KB/database_triples_831.json'), "r") as f:
    #     data = json.load(f)
    # random.shuffle(data)

    # # # TODO: triple loss这一块还没有测试集
    # train_data = data[0:int(len(data)/args.batch_size)*args.batch_size]

    # with open(osp.join(data_path, 'KG_data_tiny_831.json'),"w") as fp:
    #     json.dump(data[:1000], fp)
    kg_data_name = args.kg_data_name
    with open(osp.join(data_path, f'{kg_data_name}.json'), "r") as fp:
        train_data = json.load(fp)
    # pdb.set_trace()
    # 124169
    # 128482
    # train_data = train_data[:124168]
    # train_data = train_data[:1000]
    train_set = KGDataset(train_data)
    if args.rank == 0:
        logger.info("[End] Loading KG dataset...")
    return train_set, train_data


def _torch_collate_batch(examples, tokenizer, max_length=None, pad_to_multiple_of=None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    # are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    # if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
    #     return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.

    if max_length is None:
        pdb.set_trace()
        max_length = max(x.size(0) for x in examples)

    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0]:] = example

    return result


def load_order_data(logger, args):
    if args.rank == 0:
        logger.info("[Start] Loading Order dataset...")

    data_path = args.data_path
    if len(args.order_test_name) > 0:
        data_name = args.order_test_name
    else:
        data_name = args.order_data_name
    tmp = osp.join(data_path, f'{data_name}.json')
    if osp.exists(tmp):
        dp = tmp
    else:
        dp = osp.join(data_path, 'downstream_task', f'{data_name}.json')
    assert osp.exists(dp)
    with open(dp, "r") as fp:
        data = json.load(fp)
    # data = data[:2000]
    # pdb.set_trace()
    train_test_split = int(args.train_ratio * len(data))

    mid_split = int(train_test_split / 2)
    mid = int(len(data) / 2)
    # random.shuffle(x)
    # 训练/测试期间不应该打乱
    # train_data = data[0: train_test_split]
    # test_data = data[train_test_split: len(data)]

    # test_data = data[0: train_test_split]
    # train_data = data[train_test_split: len(data)]

    # 特殊分类 默认前一半和后一半对称
    test_data = data[0: mid_split] + data[mid: mid + mid_split]
    train_data = data[mid_split: mid] + data[mid + mid_split: len(data)]

    # pdb.set_trace()
    test_set = None
    train_set = OrderDataset(train_data)
    if len(test_data) > 0:
        test_set = OrderDataset(test_data)
    if args.rank == 0:
        logger.info("[End] Loading Order dataset...")
    return train_set, test_set, train_test_split


class Collator_order(object):
    # 输入一个batch的数据，合并order后面再解耦
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.text_maxlength = args.maxlength
        self.args = args
        # 每一个pair中包含的数据数量
        self.order_num = args.order_num
        self.p_label, self.n_label = smooth_BCE(args.eps)

    def __call__(self, batch):
        # 输入数据按顺序堆叠, 间隔拆分
        #
        # 编码然后输出
        output = []
        for item in range(self.order_num):
            output.extend([dat[0][0][item] for dat in batch])
        # label smoothing

        labels = [1 if dat[0][1][0] == 2 else self.p_label if dat[0][1][0] == 1 else self.n_label for dat in batch]
        batch = self.tokenizer.batch_encode_plus(
            output,
            padding='max_length',
            max_length=self.text_maxlength,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
            add_special_tokens=False
        )
        # torch.tensor()
        return batch, torch.FloatTensor(labels)


def smooth_BCE(eps=0.1):   # eps 平滑系数  [0, 1]  =>  [0.95, 0.05]
    # return positive, negative label smoothing BCE targets
    # positive label= y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
    # y_true=1  label_smoothing=eps=0.1
    return 1.0 - 0.5 * eps, 0.5 * eps
