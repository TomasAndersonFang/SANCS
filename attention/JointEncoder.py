#!/usr/bin/env python
# encoding: utf-8
"""
@author: Tomas S. Fang
@contact: fangsen1996@gmail.com
@software: PyCharm
@file: Joint_Encoder.py
@time: 2020/9/23 12:47
"""


from abc import ABC
import torch
import torch.nn as nn
from attention.JointEncoderLayer import JointEncoderLayer
from utils import PositionlEncoding, get_pad_mask


class JointEncoder(nn.Module, ABC):
    """
    Encoder composed with stack encoder layer.
    """

    def __init__(self, n_heads, d_k, d_v, d_model, d_ffn, dropout=0.1):
        super(JointEncoder, self).__init__()

        self.layer = JointEncoderLayer(d_model, d_ffn, n_heads, d_k, d_v)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.linear = nn.Linear(d_model, d_model)

    def forward(self, repr1, repr2):

        repr_output = self.layer(repr1, repr2, padding_mask=None)

        repr_output = self.linear(repr_output)
        repr_output = self.layer_norm(repr_output)

        return repr_output


if __name__ == "__main__":
    input1 = torch.randint(0, 10, (16, 20, 128)).float()
    input2 = torch.randint(0, 10, (16, 20, 128)).float()
    enc = JointEncoderLayer(128, 256, 8, 16, 16)
    output = enc(input1, input2)
    print(output.size())

