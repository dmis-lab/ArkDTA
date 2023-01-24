import torch
from torch import nn, optim
import numpy as np
from typing import Any, Callable, List, Tuple, Union

class DDP_BCELoss(nn.Module):
    def __init__(self, rank):
        super(DDP_BCELoss, self).__init__()
        self.criterion = nn.BCELoss().to(rank)

    def forward(self, pred, label):
        loss = self.criterion(pred, label)

        return loss

class Masked_BCELoss(nn.Module):
    def __init__(self, rank):
        super(Masked_BCELoss, self).__init__()
        #self.criterion = nn.BCELoss(reduce=False).to(rank)
        self.criterion = nn.BCELoss(reduction='none').to(rank)

    def forward(self, pred, label, pairwise_mask, vertex_mask, seq_mask):
        batch_size = pred.size(0)
        loss_all = self.criterion(pred, label)
        loss_mask = torch.matmul(vertex_mask.unsqueeze(2), seq_mask.unsqueeze(1))*pairwise_mask
        loss = torch.sum(loss_all*loss_mask) / torch.sum(pairwise_mask).clamp(min=1e-10)

        return loss

class Masked_BCELossLesser(nn.Module):
    def __init__(self, rank):
        super(Masked_BCELossLesser, self).__init__()
        #self.criterion = nn.BCELoss(reduce=False).to(rank)
        self.criterion = nn.BCELoss(reduction='none').to(rank)

    def forward(self, pred, label, mask):

        return torch.sum(self.criterion(pred, label)*mask) / torch.sum(mask).clamp(min=1e-10)

class Masked_NLLLoss(nn.Module):
    def __init__(self, rank, weight=[]):
        super(Masked_NLLLoss, self).__init__()
        if len(weight) == 0:
            #self.criterion = nn.NLLLoss(reduce=False).to(rank)
            self.criterion = nn.NLLLoss(reduction='none').to(rank)
        else:
            #self.criterion = nn.NLLLoss(reduce=False, weight=torch.cuda.FloatTensor(weight)).to(rank)
            self.criterion = nn.NLLLoss(reduction='none', weight=torch.cuda.FloatTensor(weight)).to(rank)

    def forward(self, pred, label, mask):
        pred  = (pred.view(pred.size(0)*pred.size(1),pred.size(2))+0.1).log()
        label = label.view(-1)

        mask  = mask.view(-1)
        loss  = torch.sum(self.criterion(pred, label)*mask) / torch.sum(mask).clamp(min=1e-10)

        return loss  

class Masked_CrossEntropyLoss(nn.Module):
    def __init__(self, rank, weight=[]):
        super(Masked_CrossEntropyLoss, self).__init__()
        if len(weight) == 0:
            #self.criterion = nn.CrossEntropyLoss(reduce=False).to(rank)
            self.criterion = nn.CrossEntropyLoss(reduction='none').to(rank)
        else:
            #self.criterion = nn.CrossEntropyLoss(reduce=False, weight=torch.cuda.FloatTensor(weight)).to(rank)
            self.criterion = nn.CrossEntropyLoss(reduction='none', weight=torch.cuda.FloatTensor(weight)).to(rank)

    def forward(self, pred, label, mask):
        pred  = pred.view(pred.size(0)*pred.size(1),pred.size(2))
        label = label.view(-1)

        mask  = mask.view(-1)
        loss  = torch.sum(self.criterion(pred, label)*mask) / torch.sum(mask).clamp(min=1e-10)

        return loss

class Masked_SNNLoss(nn.Module):
    def __init__(self, temp_scalar=10.0, metric='euclidean', epsilon=1e-8):
        super(Masked_SNNLoss, self).__init__()
        self.tau = temp_scalar
        self.met = metric
        self.eps = epsilon

    def forward(self, cf, rf, rfm, arG):
        '''
            receives cf  (b x 1  x d)
            receives rf  (b x n2 x d)
            receives rfm (b x n2)
            receives arG (b x n1 x n2)

            returns af (b x n x d)
            returns cf (b x d)
        '''
        cf_repeated = cf.repeat(1, rf.size(1), 1)

        # metric learning 
        if self.met == 'euclidean':
            all_terms = -1. * (cf_repeated - rf).abs().pow(2).sum(2).sqrt()
        elif self.met == 'dotproduct':
            all_terms = cf_repeated * rf
        elif self.met == 'cosdistance':
            cos = nn.CosineSimilarity(dim=2)
            all_terms = cos(cf_repeated, rf)
        else:
            raise 

        # exponential with tau
        all_terms = (all_terms / self.tau).exp()

        # masking out
        all_terms = all_terms * rfm

        # getting positives 
        all_positives = all_terms * (arG.sum(1) > 0)

        # soft nearest neighbor loss
        snn_loss = ((all_terms.sum(1) + self.eps).log() - (all_positives.sum(1) + self.eps).log())
        snn_loss = snn_loss * (arG.sum(1).sum(1) > 0.) 
        snn_loss = snn_loss.sum() / (arG.sum(1).sum(1) > 0.).sum()

        return snn_loss

class Masked_ContrastiveEuclidean(nn.Module):
    def __init__(self, rank):
        super(Masked_ContrastiveEuclidean, self).__init__()

    def forward(self, pred, label, pairwise_mask, vertex_mask, seq_mask):
        vertex_mask = vertex_mask.unsqueeze(2)
        seq_mask = seq_mask.unsqueeze(1)
        loss_mask = torch.matmul(vertex_mask, seq_mask) * pairwise_mask
        loss = torch.sum(loss_all * loss_mask) / \
            torch.sum(pairwise_mask).clamp(min=1e-10)

        return loss

class DummyScheduler:
    def __init__(self):
        x = 0
    def step(self):
        return 

def numpify(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, list):
        return [element.detach().cpu().numpy() for element in tensor]
    else:
        return tensor
