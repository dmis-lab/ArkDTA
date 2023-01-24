import torch
from torch import nn
from .base import *
from torch_geometric.nn.dense import DenseGINConv
import pickle
from .set_transformer import * 
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.nn import global_add_pool
torch.autograd.set_detect_anomaly(True)

'''
    Using Cross-Attention Mechanism,
    Q: Compound-related element-wise/pooled feature(s)
    K: Protein-related element-wise features
    V: Protein-related element-wise features, same as K
'''

def load_complex_decoder(args):
    analysis_mode = args.analysis_mode
    h = args.arkdta_hidden_dim 
    K = args.arkdta_num_heads
    S = args.arkdta_num_seeds
    T = args.arkdta_topk_pool
    A = args.arkdta_attention_option
    
    if args.arkdta_complex_decoder == 'None':
        return Decoder()
    elif args.arkdta_complex_decoder == 'PMA.Residue':
        return DecoderPMA_Residue(h, K, S, A, analysis_mode)
    else:
        raise


class Decoder(nn.Module):
    def __init__(self, analysis_mode):
        super(Decoder, self).__init__()
        self.output_representations  = []
        self.query_representations   = []
        self.kvpair_representations  = []
        self.attention_weights       = []

        if analysis_mode: self.register_forward_hook(store_decoder_representations)


    def show(self):
        print("Number of Saved Numpy Arrays: ", len(self.representations))
        for i, representation in enumerate(self.representations):
            print(f"Shape of {i}th Numpy Array: ", representation.shape)

        return self.representations

    def flush(self):
        del self.representations 
        self.representations = []

    def release_qk(self):

        return None

    def forward(self, **kwargs):

        return kwargs['X'], kwargs['X'], kwargs['residue_features'], None

class DecoderPMA_Residue(Decoder):
    def __init__(self, h: int, num_heads: int, num_seeds: int, attn_option: str, analysis_mode: bool):
        super(DecoderPMA_Residue, self).__init__(analysis_mode)
        # Aggregate the Residues into Residue Regions
        pma_args = (h, num_seeds, num_heads, RFF(h), attn_option, False, analysis_mode, False)
        self.decoder = PoolingMultiheadAttention(*pma_args)
        # Model Region-Region Interaction through Set Attention
        sab_depth = 0 if num_seeds < 4 else int((num_seeds//2) ** 0.5)
        sab_args = (h,            num_heads, RFF(h), attn_option, False, analysis_mode, True)
        self.pairwise = nn.ModuleList([SetAttentionBlock(*sab_args) for _ in range(sab_depth)])
        # Concat, then reduce into h-dimensional Set Representation
        self.aggregate = nn.Linear(h*num_seeds, h)

        self.apply(initialization)

    def forward(self, **kwargs):
        residue_features = kwargs['residue_features']
        residue_masks    = kwargs['residue_masks']
        
        output, attention = self.decoder(residue_features, residue_masks)
        for sab in self.pairwise: output, _ = sab(output)
        b, n, d = output.shape
        output = self.aggregate(output.view(b, n*d))

        return output, None, residue_features, attention