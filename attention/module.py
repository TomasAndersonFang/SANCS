#!/usr/bin/env python
# encoding: utf-8
"""
@author: Tomas S. Fang
@contact: fangsen1996@gmail.com
@software: PyCharm
@file: module.py
@time: 2020/9/8 20:42
"""


from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module, ABC):
    """
    Self-attention.
    """

    def __init__(self, factor, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask):
        attn_weight = torch.matmul(q, k.transpose(2, 3)) / self.factor

        if attn_mask is not None:
            attn_weight = attn_weight.masked_fill(attn_mask == 0, -1e9)

        attn_weight = self.dropout(F.softmax(attn_weight, dim=-1))

        context = torch.matmul(attn_weight, v)

        return context
