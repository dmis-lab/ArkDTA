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