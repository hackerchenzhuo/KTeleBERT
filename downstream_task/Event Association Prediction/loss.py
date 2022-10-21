import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PositiveUnlabledLoss(nn.Module):
    def __init__(self, class_prior=0.1):
        super(PositiveUnlabledLoss, self).__init__()
        # self.loss_fn = nn.CrossEntropyLoss()
        self.class_prior = class_prior


    def forward(self, preds, targets):
        pos_idxs = torch.nonzero(targets).reshape(-1,)
        neg_idxs = torch.nonzero(targets==0).reshape(-1)
        pos_preds = torch.index_select(preds, dim=0, index=pos_idxs)
        neg_preds = torch.index_select(preds, dim=0, index=neg_idxs)
        """
        loss_pp = F.binary_cross_entropy_with_logits(pos_preds, torch.ones_like(pos_preds, requires_grad=False))
        loss_um = F.binary_cross_entropy_with_logits(neg_preds, torch.zeros_like(neg_preds, requires_grad=False))
        loss_pm = F.binary_cross_entropy_with_logits(pos_preds, torch.zeros_like(pos_preds, requires_grad=False))
        """
        loss_pp = -torch.mean(F.logsigmoid(pos_preds[:, 1]))
        loss_um = -torch.mean(F.logsigmoid(-neg_preds[:, 1]))
        loss_pm = -torch.mean(F.logsigmoid(-pos_preds[:, 1]))
        if loss_um > self.class_prior * loss_pm:
            return self.class_prior * loss_pp - self.class_prior * loss_pm + loss_um
        else:
            return self.class_prior * loss_pp



class ContrastiveLoss(nn.Module):
    def __init__(self, temprature=2):
        super().__init__()
        self.temperature = 2
    

    def forward(self, pos_key, pos_value, neg_value):
        """
        pov_key & pos_value: n * d
        neg_value: m * d
        """
        n, m, d = pos_key.shape[0], neg_value.shape[0], neg_value.shape[1]
        pos_sim = F.cosine_similarity(pos_key, pos_value) # (n, )
        neg_key = pos_key.unsqueeze(1).expand(n, m, d)
        neg_value = neg_value.unsqueeze(0).expand(n, m, d)
        neg_sim = F.cosine_similarity(neg_key, neg_value)
        pos_loss = -torch.mean(pos_sim / self.temperature)
        neg_loss = torch.mean(
            torch.log(torch.sum(torch.exp(neg_sim / self.temperature), dim=-1)), dim=-1
        )
        loss = pos_loss + neg_loss
        return loss


if __name__ == "__main__":
    n, m, d = 10, 20, 100
    p_key = torch.randn((n, d))
    p_value = torch.randn((n, d))
    n_value = torch.randn((m, d))
    c_loss = ContrastiveLoss(temprature=2)
    constrast_loss = c_loss(p_key, p_value, n_value)
    print(constrast_loss)
