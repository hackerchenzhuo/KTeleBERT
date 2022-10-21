import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, token_num, emb_dim):
        super(TextEncoder, self).__init__()
        self.dim = emb_dim
        self.num = token_num
        self.token_embeddings = nn.Embedding(self.num, self.dim)
        

    def forward(self, data):
        text_feature = [torch.mean(self.token_embeddings(torch.LongTensor(i)), dim=0) for i in data]
        return torch.tensor([item.cpu().detach().numpy() for item in text_feature])


class GraphEncoder(nn.Module):
    def __init__(self, node_num, emb_dim):
        super(GraphEncoder, self).__init__()
        self.num = node_num
        self.dim = emb_dim
        self.node_embeddings = nn.Embedding(self.num, self.dim)
        

    def forward(self, data):
        # bs * node_dim
        graph_feature = [torch.mean(self.node_embeddings(torch.LongTensor(i)), dim=0) for i in data]
        graph_feature = torch.stack(graph_feature, dim=0)
        return graph_feature


class TimeEncoder(nn.Module):
    def __init__(self, out_dim):
        super(TimeEncoder, self).__init__()
        self.linear = nn.Linear(1, out_dim)
    
    def forward(self, time_data):
        time_data = time_data.to(torch.float32)
        time_emb = F.relu(self.linear(time_data.reshape(-1, 1)))
        return time_emb
