# -*- coding: UTF-8 -*-

import torch
from torch import nn

# https://github.com/Mikoto10032/AutomaticWeightedLoss/blob/master/AutomaticWeightedLoss.py
class AutomaticWeightedLoss(nn.Module):
    # '''
    # automatically weighted multi-task loss
    # Params£º
    #     num: int£¬the number of loss
    #     x: multi-task loss
    # Examples£º
    #     loss1=1
    #     loss2=2
    #     awl = AutomaticWeightedLoss(2)
    #     loss_sum = awl(loss1, loss2)
    # '''
    def __init__(self, num=2, args=None):
        super(AutomaticWeightedLoss, self).__init__()
        if args is None or args.use_awl:
            params = torch.ones(num, requires_grad=True)
            self.params = torch.nn.Parameter(params)
        else:
            params = torch.ones(num, requires_grad=False)
            self.params = torch.nn.Parameter(params, requires_grad=False)
            
        

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum