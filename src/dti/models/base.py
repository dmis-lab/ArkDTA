import torch
import torch.nn as nn

@torch.no_grad()
def store_representations(self, input, output):
    print(output)
    self.representations.append(output.detach().cpu().numpy())
    return

@torch.no_grad()
def store_atomwise_resiwise_interactions(self, input, output):
    self.atomwise_representations.append(output[-3].detach().cpu().numpy())
    self.resiwise_representations.append(output[-2].detach().cpu().numpy())
    self.interactions.append(output[-1].detach().cpu().numpy())
    return

@torch.no_grad()
def store_atomwise_compound_representations(self, input, output):
    self.atomwise_representations.append(output[-2].detach().cpu().numpy())
    self.compound_representations.append(output[-1].detach().cpu().numpy())
    return

@torch.no_grad()
def store_atomwise_resiwise_compound_protein_representations(self, input, output):
    self.atomwise_representations.append(output[-4].detach().cpu().numpy())
    self.resiwise_representations.append(output[-3].detach().cpu().numpy())
    self.compound_representations.append(output[-2].detach().cpu().numpy())
    self.protein_representations.append(output[-1].detach().cpu().numpy())
    return
    
@torch.no_grad()
def store_esm_representations(self, input, output):
    self.esm_representations.append(output[0].detach().cpu().numpy())
    return

@torch.no_grad()
def store_elemwise_representations(self, input, output):
    self.representations.append(output[0].detach().cpu().numpy())
    return 

@torch.no_grad()
def store_cluster_representations(self, input, output):
    self.representations.append(output[2].detach().cpu().numpy())
    return 

@torch.no_grad()
def store_decoder_representations(self, input, output):
    self.output_representations.append(output[0].detach().cpu().numpy() if output[1] else None)
    self.query_representations.append(output[1].detach().cpu().numpy() if output[1] else None)
    self.kvpair_representations.append(output[2].detach().cpu().numpy() if output[1] else None)
    self.attention_weights.append(output[3].detach().cpu().numpy() if output[1] else None)
    return

def mask_softmax(a, mask, dim=-1):
    a_max = torch.max(a, dim, keepdim=True)[0]
    a_exp = torch.exp(a - a_max)
    a_exp = a_exp * mask
    a_softmax = a_exp / (torch.sum(a_exp, dim, keepdim=True) + 1e-6)
    return a_softmax


def initialization(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.01)
# net.apply(init_weights)


class GraphDenseSequential(nn.Sequential):
    def __init__(self, *args):
        super(GraphDenseSequential, self).__init__(*args)

    def forward(self, X, adj, mask):
        for module in self._modules.values():
            try:
                X = module(X, adj, mask)
            except BaseException:
                X = module(X)

        return X


class MaskedGlobalPooling(nn.Module):
    def __init__(self, pooling='max'):
        super(MaskedGlobalPooling, self).__init__()
        self.pooling = pooling

    def forward(self, x, adj, masks):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            # print(x, adj, masks)
        masks = masks.unsqueeze(2).repeat(1, 1, x.size(2))
        if self.pooling == 'max':
            x[masks == 0] = -99999.99999
            x = x.max(1)[0]
        elif self.pooling == 'add':
            x = x.sum(1)
        else:
            print('Not Implemented')

        return x


class MaskedMean(nn.Module):
    def __init__(self):
        super(MaskedMean, self).__init__()

    def forward(self, X, m):
        if isinstance(m, torch.Tensor):
            X = X * m.unsqueeze(2) 

        return X.mean(1)


class MaskedMax(nn.Module):
    def __init__(self):
        super(MaskedMax, self).__init__()

    def forward(self, X, m):
        if isinstance(m, torch.Tensor):
            X = X * m.unsqueeze(2)

        return torch.max(X, 1)[0]


class MaskedSum(nn.Module):
    def __init__(self):
        super(MaskedSum, self).__init__()

    def forward(self, X, m):
        if isinstance(m, torch.Tensor):
            X = X * m.unsqueeze(2) 


        return X.sum(1)


class MaskedScaledAverage(nn.Module):
    def __init__(self):
        super(MaskedScaledAverage, self).__init__()

    def forward(self, X, m):
        if isinstance(m, torch.Tensor):
            X = X * m.unsqueeze(2) 

        return X.sum(1) / (m.sum(1)**0.5).unsqueeze(1)
