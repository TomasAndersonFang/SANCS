#!/usr/bin/env python
# encoding: utf-8
"""
@author: Tomas S. Fang
@contact: fangsen1996@gmail.com
@software: PyCharm
@file: Encoder.py
@time: 2020/9/15 16:48
"""


from abc import ABC
import torch
import torch.nn as nn
from attention.EncoderLayer import EncoderLayer
from utils import PositionlEncoding, get_pad_mask


class Encoder(nn.Module, ABC):
    """
    Encoder composed with stack encoder layer.
    """

    def __init__(self, vocab_size, d_word_dim, n_layers, n_heads, d_k, d_v,
                 d_model, d_ffn, pad_idx, dropout=0.1, n_position=100):
        super(Encoder, self).__init__()

        self.word_emb = nn.Embedding(vocab_size, d_word_dim, padding_idx=pad_idx)
        self.pos_enc = PositionlEncoding(d_word_dim, n_position=n_position)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_ffn, n_heads, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.linear = nn.Linear(d_model, d_model)

    def forward(self, inputs):

        padding_mask = get_pad_mask(inputs, pad_idx=self.pad_idx)
        enc_output = self.dropout(self.pos_enc(self.word_emb(inputs)))

        for enc_layer in self.layers:
            enc_output = enc_layer(enc_output, padding_mask)

        enc_output = self.linear(enc_output)
        enc_output = self.layer_norm(enc_output)

        return enc_output


if __name__ == "__main__":
    inputs1 = torch.randint(0, 10, (16, 20))
    enc = Encoder(1000, 64, 6, 8, 8, 8, 64, 256, 1)
    output = enc(inputs1)
    print(output.size())