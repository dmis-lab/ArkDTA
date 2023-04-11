import torch
from torch import nn
from .base import *
import pickle
import esm
from .set_transformer import * 
torch.autograd.set_detect_anomaly(True)

def load_residue_encoder(args):
    h = args.arkdta_hidden_dim 
    d - args.hp_dropout_rate
    C = args.arkdta_cnn_depth
    K = args.arkdta_kernel_size
    esm_model = args.arkdta_esm_model
    esm_freeze = args.arkdta_esm_freeze

    analysis_mode = args.analysis_mode

    if args.arkdta_residue_encoder == 'AminoAcidSeq.CNN':
        return AminoAcidSeqCNN(h, d, C, K, analysis_mode)
    elif args.arkdta_residue_encoder == 'Fasta.ESM':
        return FastaESM(h, esm_model, esm_freeze, analysis_mode)
    else:
        raise


class ResidueEncoder(nn.Module):
    def __init__(self):
        super(ResidueEncoder, self).__init__()
        self.representations = []

    def show(self):
        print("Number of Saved Numpy Arrays: ", len(self.representations))
        for i, representation in enumerate(self.representations):
            print(f"Shape of {i}th Numpy Array: ", representation.shape)

        return self.representations

    def flush(self):
        del self.representations 
        self.representations = []


class AminoAcidSeqCNN(ResidueEncoder):
    def __init__(self, h: int, d: float, cnn_depth: int, kernel_size: int, analysis_mode: bool):
        super(AminoAcidSeqCNN, self).__init__()
        self.encoder = nn.ModuleList([nn.Sequential(nn.Linear(21, h), # Warning
                                      nn.Dropout(d), 
                                      nn.LeakyReLU(),
                                      nn.Linear(h, h))])
        for _ in range(cnn_depth):
            self.encoder.append(nn.Conv1d(h, h, kernel_size, 1, (kernel_size-1)//2))

        self.representations = []
        if analysis_mode: self.register_forward_hook(store_representations)
        self.apply(initialization)

    def forward(self, **kwargs):
        X = kwargs['aaseqs']
        for i, module in enumerate(self.encoder):
            if i == 1: X = X.transpose(1,2)
            X = module(X)
        X = X.transpose(1,2)

        return X


class FastaESM(ResidueEncoder):
    def __init__(self, h: int, esm_model: str, esm_freeze: bool, analysis_mode: bool):
        super(FastaESM, self).__init__()
        self.esm_version = 2 if 'esm2' in esm_model else 1
        if esm_model == 'esm1b_t33_650M_UR505':
            self.esm, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            self.layer_idx, self.emb_dim = 33, 1024
        elif esm_model == 'esm1_t12_85M_UR505':
            self.esm, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
            self.layer_idx, self.emb_dim = 12, 768
        elif esm_model == 'esm2_t6_8M_UR50D':
            self.esm, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.layer_idx, self.emb_dim = 6, 320
        elif esm_model == 'esm2_t12_35M_UR50D':
            self.esm, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            self.layer_idx, self.emb_dim = 12, 480
        elif esm_model == 'esm2_t30_150M_UR50D':
            self.esm, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            self.layer_idx, self.emb_dim = 30, 640
        else:
            raise
        self.batch_converter = alphabet.get_batch_converter()
        if esm_freeze == 'True':
            for p in self.esm.parameters():
                p.requires_grad = False
        assert h == self.emb_dim, f"The hidden dimension should be set to {self.emb_dim}, not {h}"
        self.representations = []
        if analysis_mode: self.register_forward_hook(store_elemwise_representations)
        
    def esm1_pooling(self, embeddings, masks):

        return embeddings[:,1:,:].sum(1) / masks[:,1:].sum(1).view(-1,1)

    def esm2_pooling(self, embeddings, masks):

        return embeddings[:,1:-1,:].sum(1) / masks[:,1:-1].sum(1).view(-1,1)

    def forward(self, **kwargs):
        fastas = kwargs['fastas']
        _, _, tokenized = self.batch_converter(fastas)
        tokenized = tokenized.cuda()
        if self.esm_version == 2: masks = torch.where(tokenized > 1,1,0).float()
        else: masks = torch.where((tokenized > 1) & (tokenized!=32),1,0).float()

        embeddings = self.esm(tokenized, repr_layers=[self.layer_idx], return_contacts=True)
        logits     = embeddings["logits"].sum()
        contacts   = embeddings["contacts"].sum()
        attentions = embeddings["attentions"].sum()
        embeddings = embeddings["representations"][self.layer_idx]


        assert masks.size(0) == embeddings.size(0), f"Batch sizes of masks {masks.size(0)} and {embeddings.size(0)} do not match."
        assert masks.size(1) == embeddings.size(1), f"Lengths of masks {masks.size(1)} and {embeddings.size(1)} do not match."

        if self.esm_version == 2: 
            return [embeddings[:,1:-1,:], masks[:,1:-1], logits+contacts+attentions, self.esm2_pooling(embeddings, masks)]
        else:
            return [embeddings[:,1:,:],   masks[:,1:],   logits+contacts+attentions, self.esm1_pooling(embeddings, masks)]