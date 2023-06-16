import torch
from torch.nn.functional import one_hot

import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score


class NbsLoss(torch.nn.Module):
    def __init__(self, reduction='mean',
                base_loss = torch.nn.BCEWithLogitsLoss(reduction='none')):
        super().__init__()
        self.reduction = reduction
        self.base_loss = base_loss

    def forward(self, input, target, w=None):
        out = self.base_loss(input, target)
        if w is not None:
            out = out * w
        if self.reduction == 'mean':
            return out.mean()
        elif self.reduction == 'sum':
            return out.sum()
        else:
            return out

class Accuracy(torch.nn.Module):
    def __init__(self, reduction='mean', nlabels=5):
        super().__init__()
        self.reduction = reduction
        self.nlabels = nlabels

    def forward(self, input, target):

        input = torch.tensor(input)
        target = torch.tensor(target)

        if self.nlabels == 1:
            if input.ndim == 3:
                pred = input.mean(0).gt(.5).type_as(target).squeeze(-1)
            else:
                input = torch.tensor(input)
                pred = input.gt(.5).type_as(target)
        else:
            pred = input.argmax(1)

        acc = pred == target
        if self.reduction == 'mean':
            acc = acc.float().mean()
        elif self.reduction == 'sum':
            acc = acc.float().sum()
        return acc

class ConfusionMatrix(torch.nn.Module):
    def __init__(self, nlabels=5):
        super().__init__()
        self.nlabels = nlabels

    def forward(self, input, target):

        input = torch.tensor(input)
        target = torch.tensor(target)

        if self.nlabels == 1:
            if input.ndim == 3:
                pred = input.mean(0).gt(.5).type_as(target).squeeze(-1)
            else:
                pred = input.gt(.5).type_as(target)
        else:
            pred = input.argmax(1)

        cm = torch.zeros([self.nlabels, 4]).cuda()
        for l in range(self.nlabels):
            if self.nlabels == 1:
                _pred = pred.eq(1).float()
                _label = target.eq(1).float()
            else:
                _pred = pred.eq(l).float()
                _label = target.eq(l).float()

            _cm = _pred * 2 - _label

            tp = _cm.eq(1).float().sum()
            tn = _cm.eq(0).float().sum()
            fp = _cm.eq(2).float().sum()
            fn = _cm.eq(-1).float().sum()

            print(tp, tn, fp, fn)
            for j, j_ in zip(cm[l], [tp, tn, fp, fn]):
                j += j_

        return cm