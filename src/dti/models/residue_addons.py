import torch
from torch import nn
from .base import *
from torch_geometric.nn.dense import DenseGINConv
import pickle
from .set_transformer import * 
torch.autograd.set_detect_anomaly(True)

def load_residue_addon(args):
    h             = args.arkdta_hidden_dim 
    analysis_mode = args.analysis_mode
    L             = args.arkdta_sab_depth
    K             = args.arkdta_num_heads
    A             = args.arkdta_attention_option

    if args.arkdta_residue_addon == 'None':
        return ResidueAddOn()
    elif args.arkdta_residue_addon == 'ARKMAB':
        return ARKMAB(h, K, A, analysis_mode)
    else:
        raise


class ResidueAddOn(nn.Module):
    def __init__(self):
        super(ResidueAddOn, self).__init__()

        self.representations = []

    def show(self):
        print("Number of Saved Numpy Arrays: ", len(self.representations))
        for i, representation in enumerate(self.representations):
            print(f"Shape of {i}th Numpy Array: ", representation.shape)

        return self.representations

    def flush(self):
        del self.representations 
        self.representations  = []

    def forward(self, **kwargs):
        X, Xm = kwargs['X'], kwargs['Xm']

        return X, Xm

class ARKMAB(ResidueAddOn):
    def __init__(self, h: int, num_heads: int, attn_option: str, analysis_mode: bool):
        super(ARKMAB, self).__init__()
        pmx_args      = (h, num_heads, RFF(h), attn_option, False, analysis_mode, False)
        self.pmx      = PoolingMultiheadCrossAttention(*pmx_args)
        self.inactive = nn.Parameter(torch.randn(1, 1, h))
        self.fillmask = nn.Parameter(torch.ones(1,1), requires_grad=False)

        self.representations = []
        if analysis_mode: pass
        self.apply(initialization)

    def forward(self, **kwargs):
        '''
            X:  batch size x residues x H
            Xm: batch size x residues x H
            Y:  batch size x ecfpsubs x H
            Ym: batch size x ecfpsubs x H
        '''
        X, Xm = kwargs['residue_features'], kwargs['residue_masks']
        Y, Ym = kwargs['ligelem_features'], kwargs['ligelem_masks']
        pseudo_substructure = self.inactive.repeat(X.size(0),1,1)
        pseudo_masks        = self.fillmask.repeat(X.size(0),1)

        Y  = torch.cat([Y,  pseudo_substructure], 1)
        Ym = torch.cat([Ym, pseudo_masks], 1) 

        X, attention = self.pmx(Y=Y, Ym=Ym, X=X, Xm=Xm)

        return X, Xm, attention 