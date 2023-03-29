import torch
from torch import nn
from .base import *
import pickle
from .SetTransformer import * 
torch.autograd.set_detect_anomaly(True)

def load_ligelem_pooler(args):

    return args