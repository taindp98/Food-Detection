from torch import nn
import torch
from torch.nn import functional as F

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7, reduction = 'none'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss