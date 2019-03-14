"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F

from time import time as T
from util import PosEmb
from util import mypr
from layers import Conv


class QANet(nn.Module):

    def __init__(self, word_vectors, char_vectors, hidden_size=128, drop_prob=0.1, b1=1, b2=7, heads=8, device=None):
        super(QANet, self).__init__()

        self.drop_prob = drop_prob
        pos_emb = PosEmb(450, hidden_size)
        self.embed = layers.QANetEmbedding(word_vectors, char_vectors, drop_prob)
        
        infeatures = word_vectors.size(-1) + char_vectors.size(-1)
        self.emb_enc = layers.QANetEncoderLayer(infeatures=infeatures, hidden_size=hidden_size, conv_layers=4,\
                                         blocks=b1, kernel=7, heads=heads, pos_emb=pos_emb)

        self.att = layers.BiDAFAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        self.model_enc = layers.QANetEncoderLayer(infeatures=4*hidden_size, hidden_size=hidden_size, conv_layers=2, \
                                        blocks=b2, kernel=5, heads=heads, pos_emb=pos_emb)

        self.out = layers.QANetOutputLayer(hidden_size=hidden_size)

        self.drop = nn.Dropout(self.drop_prob)

        self.reduc = Conv(4*hidden_size, hidden_size, 5)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        st = T()
        cemb = self.embed(cw_idxs, cc_idxs) #(batch_size, c_len, p1+p2=500)
        qemb = self.embed(qw_idxs, qc_idxs) #(batch_size, q_len, p1+p2=500)
        mypr('Emb', T()-st)

        st = T()
        cemb = self.emb_enc(cemb, c_mask, use_pos_emb=True) #(batch_size, c_len, hidden_size)
        mypr('Con Emb Enc', T()-st)
        st = T()
        qemb = self.emb_enc(qemb, q_mask, use_pos_emb=True) #(batch_size, q_len, hidden_size)
        mypr('Query Emb Enc', T()-st) 

        st = T()
        att = self.att(cemb, qemb, c_mask, q_mask) #(batch_size, c_len, 4*hidden_size)
        mypr('Att', T()-st)

        att = self.drop(att)

        st = T()
        mask = c_mask
        model0 = self.model_enc(att, mask, use_pos_emb=True) #(batch_size, c_len, hidden_size)
        model1 = self.model_enc(model0, mask) #(batch_size, c_len, hidden_size)
        model2 = self.model_enc(model1, mask) #(batch_size, c_len, hidden_size)
        mypr('Model Enc', T()-st)

        
        # model1 = self.conv(att, c_mask)
        # model2 = self.conv(att, c_mask)
        # model0 = torch.zeros_like(model1)

        
        st = T()
        out = self.out(model0, model1, model2, c_mask) #(p1, p2)
        mypr('Out', T()-st)

        return out

    def conv(self, x, mask):
        return self.reduc(x, mask)

