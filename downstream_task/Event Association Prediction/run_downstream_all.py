import os
import xlwt
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
from mdaf_dataset import MDAFDataset
from load_scene_data import load_scene_data, load_tokenizer
from model import ReleMiner, ReleMinerPT
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from loss import PositiveUnlabledLoss

DATA = "new_datav4"
DICT_PATH = "./data/new_alarm_and_kpi.json"
EPOCH = 15
TEXT_DIM = 768
NODE_NUM = 32
NODE_DIM = 16
TIME_DIM = 4
BATCH_SIZE = 64
LR = 0.001
SEED = 2022
N_FOLD= 5
LOSS = "CE"


def generate_alarm_kpi_dict(alarm_and_kpi):
    count, res = 0, {}
    for alarm in alarm_and_kpi["alarm"]:
        res[alarm] = count
        count += 1
    for kpi in alarm_and_kpi["kpi"]:
        res[kpi] = count
        count += 1
    return res


def read_alarm_kpi_map():
    alarm_and_kpi = json.load(open(DICT_PATH, 'r'))
    result_dict = {}
    for alarm in alarm_and_kpi["alarm"].keys():
        result_dict[alarm] = alarm_and_kpi["alarm"][alarm]
    for kpi in alarm_and_kpi["kpi"].keys():
        result_dict[kpi] = alarm_and_kpi["kpi"][kpi]
    return result_dict


if __name__ == '__main__':
    ROOT_PATH = "./pre-trained/zyc_1020/"
    result_excel = xlwt.Workbook()
    result_sheet = result_excel.add_sheet("下游任务结果")
    count = 0
    for prt_emb_file in os.listdir(ROOT_PATH):
        print(prt_emb_file)
        if "ent.pt" in prt_emb_file:
            TEXT_DIM = 256
        else:
            TEXT_DIM = 768
        PRETRAIN_PATH = os.path.join(ROOT_PATH, prt_emb_file)
        if os.path.isdir(PRETRAIN_PATH):
            continue
        pretrained_emb = torch.load(open(PRETRAIN_PATH, 'rb'))
        alarm_and_kpi = json.load(open(DICT_PATH,  'r'))
        alarm_kpi_dict = generate_alarm_kpi_dict(alarm_and_kpi)
        torch.manual_seed(SEED)
        all_data = MDAFDataset(DATA, use_pretrain=True)
        # generated_neg = MDAFDataset('neg_data', use_pretrain=True)
        kf = KFold(n_splits=N_FOLD, shuffle=True)
        avg_acc, avg_precision, avg_recall, avg_f1 = 0, 0, 0, 0
        loss_log = []
        alarm_kpi_id_dict = read_alarm_kpi_map()
        manual_rules = []
        best_model = None
        best_f1 = 0
        for fold, (train_idx, test_idx) in enumerate(kf.split(all_data)):
            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(test_idx)
            train_data_loader = DataLoader(all_data, batch_size=BATCH_SIZE, sampler=train_sampler)
            test_data_loader = DataLoader(all_data, batch_size=12, sampler=test_sampler)
            ne_node_map, graph_for_each_scene = load_scene_data()
            tokenizer = load_tokenizer()
            model = ReleMinerPT(pretrained_emb, len(alarm_kpi_dict), TEXT_DIM, NODE_NUM, NODE_DIM, TIME_DIM)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LR)
            model.train()
            training_range = tqdm(range(EPOCH))
            loss_in_a_fold = []
            for epoch in training_range:
                epoch_loss = 0
                for (idx, batch_data) in enumerate(train_data_loader):
                    ent1 = torch.LongTensor([alarm_kpi_dict[str(ent)] for ent in batch_data["ent1"]])
                    ent2 = torch.LongTensor([alarm_kpi_dict[str(ent)] for ent in batch_data["ent2"]])
                    scene = batch_data["scene"]
                    label = batch_data["label"]
                    scene_text = [tokenizer[sce] for sce in batch_data["scene_text"]]
                    time = batch_data["time"]
                    target = label.clone().detach().requires_grad_(False)
                    scene_graph = [graph_for_each_scene[s] for s in scene]
                    preds = model(ent1, ent2, scene_graph, scene_text, time)
                    batch_loss = loss_fn(preds, target.long())
                    epoch_loss += batch_loss
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                training_range.set_description("Epoch Loss: {}".format(epoch_loss))
                loss_in_a_fold.append(epoch_loss)
            # print(loss_in_a_fold)
            """
            Test Part
            """
            model.eval()
            target, result = [], []
            for (idx, batch_data) in enumerate(test_data_loader):
                ent1 = torch.LongTensor([alarm_kpi_dict[str(ent)] for ent in batch_data["ent1"]])
                ent2 = torch.LongTensor([alarm_kpi_dict[str(ent)] for ent in batch_data["ent2"]])
                scene_text = [tokenizer[sce] for sce in batch_data["scene_text"]]
                scene = batch_data["scene"]
                label = batch_data["label"]
                for i in label:
                    target.append(i)
                scene_graph = [graph_for_each_scene[s] for s in scene]
                time = batch_data["time"]
                preds = model(ent1, ent2, scene_graph, scene_text, time)
                preds = F.softmax(preds, dim=1)
                for pred in preds:
                    if pred[0] > pred[1]:
                        result.append(0)
                    else:
                        result.append(1)
            acc = accuracy_score(y_true=target, y_pred=result)
            f1 = f1_score(y_true=target, y_pred=result)
            precision = precision_score(y_true=target, y_pred=result)
            recall = recall_score(y_true=target, y_pred=result)
            avg_acc += acc
            avg_f1 += f1
            avg_precision += precision
            avg_recall += recall
            print("================Fold {}====================".format(fold))
            print("Acc = {}".format(acc))
            print("F1 = {}".format(f1))
            print("Precision = {}".format(precision))
            print("Recall = {}".format(recall))
            if best_f1 < avg_f1:
                best_f1 = avg_f1
                best_model = model
        print("================All====================")
        print("Acc = {}".format(avg_acc / N_FOLD))
        print("F1 = {}".format(avg_f1 / N_FOLD))
        print("Precision = {}".format(avg_precision / N_FOLD))
        print("Recall = {}".format(avg_recall / N_FOLD))
        print("EPOCH = {}, TEXT_DIM = {}, NODE_DIM = {}, LR = {}, SEED = {}".format(EPOCH, TEXT_DIM, NODE_DIM, LR, SEED))
        result_sheet.write(count, 0, prt_emb_file)
        result_sheet.write(count, 1, round(avg_acc / 5 * 100, 2))
        result_sheet.write(count, 2, round(avg_precision / 5 * 100, 2))
        result_sheet.write(count, 3, round(avg_recall / 5 * 100, 2))
        result_sheet.write(count, 4, round(avg_f1 / 5 * 100, 2))
        count += 1
    result_excel.save("故障规则挖掘实验结果_zyc_1020.xls")
        