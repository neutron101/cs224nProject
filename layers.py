"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import masked_softmax
import numpy as np
from time import time as T
from util import mypr
        
class QANetEncoderLayer(nn.Module):

    def __init__(self, infeatures, conv_layers, kernel, heads, blocks, hidden_size, pos_emb, pL=.9):
        super(QANetEncoderLayer, self).__init__()

        self.blocks = nn.ModuleList([QANetEncoderBlock(hidden_size=hidden_size, \
                            conv_layers=conv_layers, kernel=kernel, heads=heads, pL=(1-pL), pos_emb=pos_emb) for _ in range(blocks)])

        self.dim_mapper = Conv(infeatures, hidden_size, 1)
        self.total_runs = blocks*conv_layers

    def forward(self, emb, mask, reduc_dim=False):

        if reduc_dim:
            sa = emb.size()
            emb = self.dim_mapper(emb, mask)
            sb = emb.size()
            assert sa[0:2] == sb[0:2]

        depth = 0
        # Move blocks forward
        for b in self.blocks:
            emb, depth = b(emb, mask, depth, self.total_runs)

        return emb


class QANetEncoderBlock(nn.Module):

    def __init__(self, hidden_size, conv_layers, kernel, heads, pL, pos_emb, dropout=0.1):
        super(QANetEncoderBlock, self).__init__()

        self.cnns = nn.ModuleList([ResBlock(Conv(hidden_size, hidden_size, kernel, depth=True), hidden_size) \
                                         for _ in range(conv_layers)])

        self.att = QANetAttBlock(hidden_size, heads=heads)
        
        self.feedforward = ResBlock(nn.Linear(hidden_size, hidden_size, True), hidden_size)
        self.pL = pL
        self.pos_emb = pos_emb


    def forward(self, emb, mask, depth, total_runs):         

        # Get positional embedding
        # p_emb = self.pos_emb.emb[0:emb.size(1), :]
        # p_emb = p_emb.unsqueeze(0)
        # p_emb = p_emb.to(emb.device)
        # emb = emb + p_emb  

        out = emb
        for c in self.cnns:
            depth += 1
            out = c(out, mask)

        out = self.att(out, mask)
        out = self.feedforward(out)
        out = F.relu(out)
        
        return out, depth


class QANetAttBlock(nn.Module):
    """docstring for QANetAttBlock"""
    def __init__(self, hidden_size, heads):
        super(QANetAttBlock, self).__init__()

        dk = hidden_size//heads
        self.heads = heads
        self.W_q = nn.ModuleList([nn.Linear(hidden_size, dk, False) for _ in range(heads)])
        self.W_k = nn.ModuleList([nn.Linear(hidden_size, dk, False) for _ in range(heads)])
        self.W_v = nn.ModuleList([nn.Linear(hidden_size, dk, False) for _ in range(heads)])

        self.W_out = nn.Linear(heads*dk, hidden_size, False)
        self.sfmax = nn.Softmax(dim=2)
        self.dkroot = np.sqrt(dk)

        self.norm = nn.LayerNorm(hidden_size)

    
    def forward(self, x, mask):
                
        x = self.norm(x)

        nmask = mask.unsqueeze(1).expand(-1, 1, -1)
        nmask = nmask.type(torch.float32)
        h = []
        for W_q, W_k, W_v in zip(self.W_q, self.W_k, self.W_v):
            h.append(self.attn(W_q(x), W_k(x), W_v(x), nmask))

        attn = self.W_out(torch.cat(h, dim=2))
        attn = attn + x
        nmask = nmask.transpose(1,2)
        attn = attn * nmask

        return attn 


    def attn(self, Q, K, V, mask):
        
        res = torch.matmul(Q, torch.transpose(K,1,2))
        res = torch.div(res, self.dkroot)
        res = self.masked_softmax(res, mask)
        attn = torch.matmul(res, V) 

        return attn


    def masked_softmax(self, logits, mask):
        masked_logits = mask * logits + (1 - mask) * -1e30
        probs = self.sfmax(masked_logits)
        return probs


class Conv(nn.Module):
    """docstring for DepthSepConv"""
    def __init__(self, in_features, hidden_size, kernel, depth=False):
        super(  Conv, self).__init__()

        self.hidden_size = hidden_size
        if depth:
            self.cnns = nn.ModuleList([ \
                nn.Conv1d(in_features, in_features, kernel, padding=kernel//2, groups=in_features), \
                nn.Conv1d(in_features, hidden_size, 1)])
        else:
            self.cnns = nn.ModuleList([nn.Conv1d(in_features, hidden_size, kernel, padding=kernel//2)])

    def forward(self, x, mask):
        nmask = mask.unsqueeze(2).type(torch.float32)
        # Optional
        # x = nmask * x
        
        out = x.permute(0,2,1)

        for c in self.cnns:
            result = c(out)
            out = result      

        out = out.permute(0, 2, 1)

        assert x.size()[0:2] == out.size()[0:2], '{} {}'.format(x.size(), out.size())
        assert out.size()[2] == self.hidden_size, '{} {}'.format(out.size()[2], self.hidden_size)

        return out


class ResBlock(nn.Module):
    """docstring for ResBlock"""
    def __init__(self, module, hidden_size):
        super(ResBlock, self).__init__()
        self.module = module
        self.norm = nn.LayerNorm(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x, *args):

        result = x
        xt = self.norm(x)
        xt = self.module(xt, *args)
        result = result + xt
            
        return result


class QANetOutputLayer(nn.Module):

    def __init__(self, hidden_size):
        super(QANetOutputLayer, self).__init__()

        self.stlinear = nn.Linear(hidden_size*2, 1, False)
        self.endlinear = nn.Linear(hidden_size*2, 1, False)

    def forward(self, M0, M1, M2, mask):
    
        M0 = torch.transpose(M0, 1,2)
        M1 = torch.transpose(M1, 1,2)
        st_merged = torch.cat([M0, M1], dim=1).transpose(1,2)

        M2 = torch.transpose(M2, 1,2)
        end_merged = torch.cat([M0, M2], dim=1).transpose(1,2)

        st_linearized = self.stlinear(st_merged)
        end_linearized = self.endlinear(end_merged)

        log_pst = masked_softmax(st_linearized.squeeze(), mask, log_softmax=True)
        log_pend = masked_softmax(end_linearized.squeeze(), mask, log_softmax=True)

        return log_pst, log_pend


class CNN(nn.Module):

    # filter size and width from the paper
    # https://arxiv.org/pdf/1611.01603.pdf
    def __init__(self, char_embeddings, filters=100, width=5):

        super(CNN, self).__init__()

        self.e_char = char_embeddings.size(1)
        self.filters = filters
        self.width = width

        self.embeddings = nn.Embedding.from_pretrained(char_embeddings, freeze=False)
        self.conv1d = nn.Conv1d(self.e_char, self.filters, self.width)
        self.relu = nn.ReLU()

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        #list of words with length m_word
        # @param x_reshaped (Tensor) - # (batch_size, sentence_length, max_word_length),
        # @returns (Tensor) - (batch_size, sentence_length, filters)

        chars = self.embeddings(chars)
        chars = chars.permute(1,0,2,3)

        #(sentence_length, batch_size, max_word_length, e_char)
        chars_size = chars.size()

        o = chars.permute(1,0,2,3)
        o = o.contiguous().view(chars_size[0]*chars_size[1], chars_size[2], chars_size[3])
        o = o.permute(0,2,1)  

        x_conv = self.conv1d(o)
        x_conv = self.relu(x_conv)
        m_word = o.size()[2]
        maxpool = nn.MaxPool1d(m_word - self.width + 1) #could be initialized once
        x_conv = maxpool(x_conv)

        x_conv = x_conv.squeeze(dim=-1).view(chars_size[1], chars_size[0], -1)

        return x_conv


class QANetEmbedding(nn.Module):
    """Embedding layer used by QANet, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, w_drop_prob=0.1, c_drop_prob=.05):
        super(QANetEmbedding, self).__init__()
 
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.embed_unk = nn.Embedding(1, word_vectors.size(1))
        
        self.proj = nn.Linear(word_vectors.size(1), word_vectors.size(1), bias=False)

        self.cemb = CNN(char_embeddings=char_vectors, filters=char_vectors.size(1))

        self.hwy = HighwayEncoder(2, word_vectors.size(1)+char_vectors.size(1))

        self.UNK = 1

        self.wdrop = nn.Dropout(w_drop_prob)
        self.cdrop = nn.Dropout(c_drop_prob)

    def forward(self, x, c):
        # get charCNN embeddings
        cemb = self.cemb(c)
        cemb = self.cdrop(cemb)

        # get word embeddings
        emb = self.get_emb(x) 
        emb = self.wdrop(emb)
        emb = self.proj(emb)

        assert cemb.size()[0:2] == emb.size()[0:2], 'emb size {}{} csize {}{}'.format(x.size(), emb.size(), c.size(), cemb.size())

        # concatenate word and char embeddings
        emb = torch.cat((emb,cemb), dim=2)
        # OPT: individual hwy first and then concatenate
        emb = self.hwy(emb) 

        return emb

    def get_emb(self, x):

        emb = self.embed(x) 
        unk_emb = self.embed_unk(torch.tensor([[0]], dtype=torch.long, device=x.device))
        
        mask = x.eq(self.UNK)
        # mask[:,0] = 0
        mask = mask.type(torch.float32)
        mask = mask.unsqueeze(2)
        emb = (1-mask)*emb + mask*unk_emb

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s
