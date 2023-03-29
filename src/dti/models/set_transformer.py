import torch
import torch.nn as nn 
from torch.nn.functional import normalize as l2
import math
import torch.nn.functional as F
import numpy as np


class DotProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, queries, keys):

        return torch.bmm(queries, keys.transpose(1, 2))

class ScaledDotProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, queries, keys):

        return torch.bmm(queries, keys.transpose(1, 2)) / (queries.size(2)**0.5)

class GeneralDotProduct(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        torch.nn.init.orthogonal_(self.W)

    def forward(self, queries, keys):

        return torch.bmm(queries @ self.W, keys.transpose(1,2))

class ConcatDotProduct(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        raise 

    def forward(self, queries, keys):

        return

class Additive(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.U = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.T = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.W = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim,1))
        torch.nn.init.orthogonal_(self.U)
        torch.nn.init.orthogonal_(self.T)

    def forward(self, queries, keys):

        return self.W(queries.unsqueeze(1)@self.U + keys.unsqueeze(2)@self.T + self.b).squeeze(-1).transpose(1,2)


class Attention(nn.Module):
    def __init__(self, similarity, hidden_dim=1024, store_qk=False):
        super().__init__()
        self.softmax = nn.Softmax(dim=2)
        self.attention_maps = []
        self.store_qk = store_qk
        self.query_vectors, self.key_vectors = None, None

        assert similarity in ['dot', 'scaled_dot', 'general_dot', 'concat_dot', 'additive']
        if similarity == 'dot':
            self.similarity = DotProduct()
        elif similarity == 'scaled_dot':
            self.similarity = ScaledDotProduct()
        elif similarity == 'general_dot':
            self.similarity = GeneralDotProduct(hidden_dim)
        elif similarity == 'concat_dot':
            self.similarity = ConcatDotProduct(hidden_dim)
        elif similarity == 'additive':
            self.similarity = Additive(hidden_dim)
        else:
            raise

    def release_qk(self):
        Q, K = self.query_vectors, self.key_vectors
        self.query_vectors, self.key_vectors = None, None
        torch.cuda.empty_cache()

        return Q, K

    def forward(self, queries, keys, qmasks=None, kmasks=None):
        if self.store_qk: 
            self.query_vectors = queries
            self.key_vectors   = keys

        if torch.is_tensor(qmasks) and not torch.is_tensor(kmasks):
            dim0, dim1 = qmasks.size(0), keys.size(1)
            kmasks = torch.ones(dim0,dim1).cuda()

        elif not torch.is_tensor(qmasks) and torch.is_tensor(kmasks):
            dim0, dim1 = kmasks.size(0), queries.size(1)
            qmasks = torch.ones(dim0,dim1).cuda()
        else:
            pass

        attention = self.similarity(queries, keys)
        if torch.is_tensor(qmasks) and torch.is_tensor(kmasks):
            qmasks = qmasks.repeat(queries.size(0)//qmasks.size(0),1).unsqueeze(2)
            kmasks = kmasks.repeat(keys.size(0)//kmasks.size(0),1).unsqueeze(2)
            attnmasks = torch.bmm(qmasks, kmasks.transpose(1, 2))
            attention = torch.clip(attention, min=-10, max=10)
            attention = attention.exp()
            attention = attention * attnmasks
            attention = attention / (attention.sum(2).unsqueeze(2) + 1e-5)
        else:
            attention = self.softmax(attention)

        return attention

@torch.no_grad()
def save_attention_maps(self, input, output):
    
    self.attention_maps.append(output.data.detach().cpu().numpy())

class MultiheadAttention(nn.Module):
    def __init__(self, d, h, sim='dot', analysis=False, store_qk=False):
        super().__init__()
        assert d % h == 0, f"{d} dimension, {h} heads"
        self.h = h
        p = d // h

        self.project_queries = nn.Linear(d, d)
        self.project_keys    = nn.Linear(d, d)
        self.project_values  = nn.Linear(d, d)
        self.concatenation   = nn.Linear(d, d)
        self.attention       = Attention(sim, p, store_qk)

        if analysis:
            self.attention.register_forward_hook(save_attention_maps)

    def release_qk(self):
        Q, K = self.attention.release_qk()

        Qb    = Q.size(0)//self.h
        Qn, Qd = Q.size(1), Q.size(2)

        Kb    = K.size(0)//self.h
        Kn, Kd = K.size(1), K.size(2)

        Q = Q.view(self.h, Qb, Qn, Qd)
        K = K.view(self.h, Kb, Kn, Kd)

        Q = Q.permute(1,2,0,3).contiguous().view(Qb, Qn, Qd*self.h) 
        K = K.permute(1,2,0,3).contiguous().view(Kb, Kn, Kd*self.h)

        return Q, K

    def forward(self, queries, keys, values, qmasks=None, kmasks=None):
        h = self.h
        b, n, d = queries.size()
        _, m, _ = keys.size()
        p = d // h

        queries = self.project_queries(queries)  # shape [b, n, d]
        keys = self.project_keys(keys)  # shape [b, m, d]
        values = self.project_values(values)  # shape [b, m, d]

        queries = queries.view(b, n, h, p)
        keys = keys.view(b, m, h, p)
        values = values.view(b, m, h, p)

        queries = queries.permute(2, 0, 1, 3).contiguous().view(h * b, n, p)
        keys = keys.permute(2, 0, 1, 3).contiguous().view(h * b, m, p)
        values = values.permute(2, 0, 1, 3).contiguous().view(h * b, m, p)

        attn_w = self.attention(queries, keys, qmasks, kmasks)  # shape [h * b, n, p]
        output = torch.bmm(attn_w, values)
        output = output.view(h, b, n, p)
        output = output.permute(1, 2, 0, 3).contiguous().view(b, n, d)
        output = self.concatenation(output)  # shape [b, n, d]

        return output, attn_w

class MultiheadAttentionExpanded(nn.Module):
    def __init__(self, d, h, sim='dot', analysis=False):
        super().__init__()
        self.project_queries = nn.ModuleList([nn.Linear(d,d) for _ in range(h)])
        self.project_keys    = nn.ModuleList([nn.Linear(d,d) for _ in range(h)])
        self.project_values  = nn.ModuleList([nn.Linear(d,d) for _ in range(h)])
        self.concatenation   = nn.Linear(h*d, d)
        self.attention       = Attention(sim, d)

        if analysis:
            self.attention.register_forward_hook(save_attention_maps)

    def forward(self, queries, keys, values, qmasks=None, kmasks=None):
        output = []
        for Wq, Wk, Wv in zip(self.project_queries, self.project_keys, self.project_values):
            Pq, Pk, Pv = Wq(queries), Wk(keys), Wv(values)
            output.append(torch.bmm(self.attention(Pq, Pk, qmasks, kmasks), Pv))

        output = self.concatenation(torch.cat(output, 1))

        return output

class EmptyModule(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, x):
        return 0.


class RFF(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.rff = nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())

    def forward(self, x):

        return self.rff(x)


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d, h, rff, similarity='dot', full_head=False, analysis=False, store_qk=False):
        super().__init__()
        self.multihead = MultiheadAttention(d, h, similarity, analysis, store_qk) if not full_head else MultiheadAttentionExpanded(d, h, similarity, analysis)
        self.layer_norm1 = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.rff = rff

    def release_qk(self):
        Q, K = self.multihead.release_qk()

        return Q, K

    def forward(self, x, y, xm=None, ym=None, layer_norm=True):
        h, a = self.multihead(x, y, y, xm, ym)
        if layer_norm:
            h = self.layer_norm1(x + h)
            return self.layer_norm2(h + self.rff(h)), a
        else:
            h = x + h
            return h + self.rff(h), a


class SetAttentionBlock(nn.Module):
    def __init__(self, d, h, rff, similarity='dot', full_head=False, analysis=False, store_qk=False):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff, similarity, full_head, analysis, store_qk)

    def release_qk(self):
        Q, K = self.mab.release_qk()

        return Q, K

    def forward(self, x, m=None, ln=True):

        return self.mab(x, x, m, m, ln)

class InducedSetAttentionBlock(nn.Module):
    def __init__(self, d, m, h, rff1, rff2, similarity='dot', full_head=False, analysis=False, store_qk=False):

        super().__init__()
        self.mab1 = MultiheadAttentionBlock(d, h, rff1, similarity, full_head, analysis, store_qk)
        self.mab2 = MultiheadAttentionBlock(d, h, rff2, similarity, full_head, analysis, store_qk)
        self.inducing_points = nn.Parameter(torch.randn(1, m, d))

    def release_qk(self):

        raise NotImplemented

    def forward(self, x, m=None, ln=True):
        b = x.size(0)
        p = self.inducing_points
        p = p.repeat([b, 1, 1])  # shape [b, m, d]
        h = self.mab1(p, x, None, m, ln)  # shape [b, m, d]

        return self.mab2(x, h, m, None, ln)

class PoolingMultiheadAttention(nn.Module):
    def __init__(self, d, k, h, rff, similarity='dot', full_head=False, analysis=False, store_qk=False):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff, similarity, full_head, analysis, store_qk)
        self.seed_vectors = nn.Parameter(torch.randn(1, k, d))
        torch.nn.init.xavier_uniform_(self.seed_vectors)

    def release_qk(self):
        Q, K = self.mab.release_qk()

        return Q, K

    def forward(self, z, m=None, ln=True):
        b = z.size(0)
        s = self.seed_vectors
        s = s.repeat([b, 1, 1])  # random seed vector: shape [b, k, d]

        return self.mab(s, z, None, m, ln)

class PoolingMultiheadCrossAttention(nn.Module):
    def __init__(self, d, h, rff, similarity='dot', full_head=False, analysis=False, store_qk=False):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff, similarity, full_head, analysis, store_qk)

    def release_qk(self):
        Q, K = self.mab.release_qk()

        return Q, K

    def forward(self, X, Y, Xm=None, Ym=None, ln=True):
        
        return self.mab(X, Y, Xm, Ym, ln)