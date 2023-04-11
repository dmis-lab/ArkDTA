import torch
from torch import nn
from .base import *
from torch_geometric.nn.dense import DenseGINConv
import pickle
from .set_transformer import * 
from transformers import RobertaModel, RobertaTokenizer
torch.autograd.set_detect_anomaly(True)

def load_ligelem_encoder(args):

    return args

class LigelemEncoder(nn.Module):
    def __init__(self):
        super(LigelemEncoder, self).__init__()
        self.representations = []

    def show(self):
        print("Number of Saved Numpy Arrays: ", len(self.representations))
        for i, representation in enumerate(self.representations):
            print(f"Shape of {i}th Numpy Array: ", representation.shape)

        return self.representations

    def flush(self):
        del self.representations 
        self.representations = []


class EcfpConverter(LigelemEncoder):
    def __init__(self, h: int, sab_depth: int, ecfp_dim: int, analysis_mode: bool):
        super(EcfpConverter, self).__init__()
        K = 4 # number of attention heads
        self.ecfp_embeddings = nn.Embedding(ecfp_dim+1, h, padding_idx=ecfp_dim)
        self.encoder     = nn.ModuleList([])
        sab_args = (h, K, RFF(h), 'general_dot', False, analysis_mode, True)
        self.encoder = nn.ModuleList([SetAttentionBlock(*sab_args) for _ in range(sab_depth)])

        self.representations = []
        if analysis_mode: self.register_forward_hook(store_elemwise_representations)
        self.apply(initialization)

    def forward(self, **kwargs):
        '''
            X : (b x d)
        '''
        ecfp_words = kwargs['ecfp_words']
        ecfp_masks = kwargs['ecfp_masks']
        ecfp_words = self.ecfp_embeddings(ecfp_words)

        for sab in self.encoder:
            ecfp_words, _ = sab(ecfp_words, ecfp_masks)

        return [ecfp_words, ecfp_masks]