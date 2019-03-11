"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F


class QANet(nn.Module):

    def __init__(self, word_vectors, char_vectors, hidden_size=128, drop_prob=0.1):
        super(QANet, self).__init__()

        self.drop_prob = drop_prob
        self.embed = layers.QANetEmbedding(word_vectors, char_vectors, drop_prob)
        
        infeatures = word_vectors.size(-1) + char_vectors.size(-1)
        self.emb_enc = layers.QANetEncoderLayer(infeatures=infeatures, hidden_size=hidden_size, conv_layers=4,\
                                         blocks=1, kernel=7,  device=word_vectors.device)

        self.att = layers.BiDAFAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        self.model_enc = layers.QANetEncoderLayer(infeatures=4*hidden_size, hidden_size=hidden_size, conv_layers=2, \
                                        blocks=7, kernel=5, device=word_vectors.device)

        self.out = layers.QANetOutputLayer(hidden_size=hidden_size)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        cemb = self.embed(cw_idxs, cc_idxs) #(batch_size, c_len, p1+p2=500)
        qemb = self.embed(qw_idxs, qc_idxs) #(batch_size, q_len, p1+p2=500)

        cemb = self.emb_enc(cemb, c_mask, use_pos_emb=True) #(batch_size, c_len, hidden_size)
        qemb = self.emb_enc(qemb, q_mask, use_pos_emb=True) #(batch_size, q_len, hidden_size)
        cemb = F.dropout(cemb, self.drop_prob, self.training)
        qemb = F.dropout(qemb, self.drop_prob, self.training)

        att = self.att(cemb, qemb, c_mask, q_mask) #(batch_size, c_len, 4*hidden_size)
        att = F.dropout(att, self.drop_prob, self.training)

        mask = torch.ones_like(c_mask, dtype=torch.int32)
        model0 = self.model_enc(att, mask, use_pos_emb=True) #(batch_size, c_len, hidden_size)
        model1 = self.model_enc(model0, mask) #(batch_size, c_len, hidden_size)
        model2 = self.model_enc(model1, mask) #(batch_size, c_len, hidden_size)
        model0 = F.dropout(model0, self.drop_prob, self.training)
        model1 = F.dropout(model1, self.drop_prob, self.training)
        model2 = F.dropout(model2, self.drop_prob, self.training)
        
        out = self.out(model0, model1, model2, c_mask) #(batch_size, c_len)

        return out