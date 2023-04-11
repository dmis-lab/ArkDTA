import torch
from torch import nn
from .base import *
torch.autograd.set_detect_anomaly(True)

def load_affinity_predictor(args):

    return args

def load_interaction_predictor(args):    

    return args

class AffinityMLP(nn.Module):
    def __init__(self, h: int):
        super(AffinityMLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(h,h), nn.Dropout(0.1), nn.LeakyReLU(), nn.Linear(h,1))

        self.apply(initialization)

    def forward(self, **kwargs):
        '''
            X:  batch size x 1 x H 
        '''
        X = kwargs['binding_complex']
        X = X.squeeze(1) if X.dim() == 3 else X 
        yhat = self.mlp(X)

        return yhat

class InteractionMLP(nn.Module):
    def __init__(self, h: int):
        super(InteractionMLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(h,h), nn.Dropout(0.1), nn.LeakyReLU(), nn.Linear(h,1), nn.Sigmoid())

        self.apply(initialization)

    def forward(self, **kwargs):
        '''
            X:  batch size x 1 x H 
        '''
        X = kwargs['binding_complex']
        X = X.squeeze(1) if X.dim() == 3 else X 
        yhat = self.mlp(X)

        return yhat