"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertModel

class Berty(BertPreTrainedModel):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).


    Args:
    """
    def __init__(self, config, word_emb_size, vocabulary, max_seq_length=512):
        super(Berty, self).__init__(config)
        self.bert = BertModel(config)

        self.qa_outputs = nn.Linear(config.hidden_size, word_emb_size)
        self.apply(self.init_bert_weights)

        self.CLS_idx = [vocabulary['[CLS]']]
        self.SEP_idx = [vocabulary['[SEP]']]
        self.word_emb_size = word_emb_size


    def forward(self, cw_idxs, qw_idxs, c_mask, q_mask):

        # The convention in BERT is:
        # (a) For sequence pairs: <query> <context>
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1

        # PAD at the end of 
        # prepend CLS
        # append SEP
        # append SEP

        CLS_idx = torch.tensor(self.CLS_idx, device=cw_idxs.device)
        SEP_idx = torch.tensor(self.SEP_idx, device=cw_idxs.device)
        max_seq_length = cw_idxs.size()[1] + qw_idxs.size()[1] + 1

        attention_masks = []
        input_words = []
        segments = []

        for sen in range(cw_idxs.shape[0]):
            qw = torch.zeros(max_seq_length, dtype=cw_idxs.dtype, device=cw_idxs.device)
            st = 0
            end = q_mask[sen].sum()
            qw[st: end] = qw_idxs[sen, 0: q_mask[sen].sum()]
            qw[0] = CLS_idx[0]
            qw[end] = SEP_idx[0]
			
            st = end+1
            end = st + c_mask[sen].sum()-1
            qw[st:end] = cw_idxs[sen, 1: c_mask[sen].sum()]
            qw[end] = SEP_idx[0]
			
            qwords = qw

            segment = torch.zeros(max_seq_length, device=cw_idxs.device, dtype=cw_idxs.dtype)
            segment[q_mask[sen].sum()+1 : q_mask[sen].sum()+1+c_mask[sen].sum()] = 1
			
            balance = (max_seq_length - end)

            attention_mask = torch.cat((torch.ones(end, dtype=cw_idxs.dtype, device=cw_idxs.device), torch.zeros(balance, dtype=cw_idxs.dtype, device=cw_idxs.device)))


            assert attention_mask.size()[0] == max_seq_length
            assert qwords.size()[0] == max_seq_length
            assert segment.size()[0] == max_seq_length

            input_words.append(qwords)
            attention_masks.append(attention_mask)
            segments.append(segment)

        input_words = torch.stack(input_words, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)
        segments = torch.stack(segments, dim=0)

        output, _ = self.bert(input_words, segments, attention_masks, output_all_encoded_layers=False)
        emb = self.qa_outputs(output)

        cws = []
        qws = []
        qlength = qw_idxs.shape[1]
        clength = cw_idxs.shape[1]

        for s in range(emb.size()[0]):
            sen = emb[s]

            qs = torch.zeros((qlength, self.word_emb_size), device=cw_idxs.device)
            qlen = q_mask[s].sum()
            qs[1:qlen,:] = sen[1:qlen,:]
            qs[0]=1

            cs = torch.zeros((clength, self.word_emb_size), device=cw_idxs.device)
            clen = c_mask[s].sum()
            cs[1:clen,:] = sen[qlen+1:qlen+clen,:]
            cs[0]=1
        
            cws.append(cs)
            qws.append(qs)

        cws = torch.stack(cws, dim=0)
        qws = torch.stack(qws, dim=0)


        return cws, qws


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


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, vocabulary, char_vectors, drop_prob, cnn_features, word_emb_size, max_seq_length, hidden_size):
        super(Embedding, self).__init__()

        self.embed = Berty.from_pretrained('bert-base-uncased', \
            cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(1)), word_emb_size=word_emb_size, vocabulary=vocabulary)
        
        self.cemb = CNN(char_embeddings=char_vectors, filters=cnn_features)
        self.drop_prob = drop_prob

        self.proj = nn.Linear(word_emb_size, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size+cnn_features)


    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs, c_mask, q_mask):
        # get charCNN embeddings
        cc_emb = self.cemb(cc_idxs)
        qc_emb = self.cemb(qc_idxs)

        embc, embq = self.embed(cw_idxs, qw_idxs, c_mask, q_mask)   # (batch_size, seq_len, embed_size)
        
        # CONTEXT
        emb = F.dropout(embc, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        # concatenate word and char embeddings
        emb = torch.cat((emb,cc_emb), dim=2)
        embc = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        # QUERY
        emb = F.dropout(embq, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        # concatenate word and char embeddings
        emb = torch.cat((emb,qc_emb), dim=2)
        embq = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return embc, embq


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


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

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


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
