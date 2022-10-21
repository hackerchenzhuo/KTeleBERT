import torch
import torch.nn as nn
import torch.nn.functional as F
from module import TextEncoder, GraphEncoder, TimeEncoder


class ReleMiner(nn.Module):
    def __init__(
        self, 
        token_num, 
        text_emb_dim, 
        node_num, 
        node_emb_dim, 
        time_dim
    ):
        super(ReleMiner, self).__init__()
        self.text_encoder = TextEncoder(token_num, text_emb_dim)
        self.graph_encoder = GraphEncoder(node_num, node_emb_dim)
        self.time_encoder = TimeEncoder(time_dim)
        self.linear = nn.Linear(3 * text_emb_dim + node_emb_dim + time_dim, 2)

    def forward(
        self, 
        text1, 
        text2, 
        graph_data, 
        scene_text,
        time_data,
        return_emb=False
    ):
        text_emb1 = self.text_encoder(text1)
        text_emb2 = self.text_encoder(text2)
        graph_emb = self.graph_encoder(graph_data)
        scene_text_emb = self.text_encoder(scene_text)
        rule_emb = torch.cat((text_emb1, text_emb2), dim=-1)
        scene_emb = torch.cat((graph_emb, scene_text_emb), dim=-1)
        time_emb = self.time_encoder(time_data)
        text_and_scene_emb = torch.cat((rule_emb, scene_emb), dim=-1)
        all_emb = torch.cat((text_and_scene_emb, time_emb), dim=-1)
        preds = self.linear(all_emb)
        if return_emb:
            return preds, all_emb
        return preds


class ReleMinerPT(nn.Module):
    def __init__(
        self,
        pretrained_emb,
        alarm_kpi_num,
        text_emb_dim, 
        node_num, 
        node_emb_dim, 
        time_dim
    ):
        super(ReleMinerPT, self).__init__()
        self.text_encoder = nn.Embedding(alarm_kpi_num, text_emb_dim).from_pretrained(pretrained_emb)
        self.graph_encoder = GraphEncoder(node_num, node_emb_dim)
        self.time_encoder = TimeEncoder(time_dim)
        self.linear = nn.Linear(2 * text_emb_dim + node_emb_dim + time_dim, 2)

    def forward(
        self, 
        text1, 
        text2, 
        graph_data, 
        scene_text,
        time_data,
        return_emb=False
    ):
        text_emb1 = self.text_encoder(text1)
        text_emb2 = self.text_encoder(text2)
        graph_emb = self.graph_encoder(graph_data)
        # scene_text_emb = self.text_encoder(scene_text)
        rule_emb = torch.cat((text_emb1, text_emb2), dim=-1)
        # scene_emb = torch.cat((graph_emb, scene_text_emb), dim=-1)
        time_emb = self.time_encoder(time_data)
        text_and_scene_emb = torch.cat((rule_emb, graph_emb), dim=-1)
        all_emb = torch.cat((text_and_scene_emb, time_emb), dim=-1)
        preds = self.linear(all_emb)
        if return_emb:
            return preds, all_emb
        return preds
