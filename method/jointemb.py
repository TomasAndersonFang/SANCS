#!/usr/bin/env python
# encoding: utf-8
"""
@author: Tomas S. Fang
@contact: fangsen1996@gmail.com
@software: PyCharm
@file: JointEmbeder.py
@time: 2020/8/13 16:04
"""

from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention.Encoder import Encoder
from attention.JointEncoder import JointEncoder
from utils import get_pad_mask


class JointEmbedder(nn.Module, ABC):
    """
    Joint embedding for code snippets and code description, and have a extra
    attention to code snippet components to build internal relationship.
    """

    def __init__(self, config):
        super(JointEmbedder, self).__init__()

        self.conf = config
        self.margin = config["margin"]

        self.name_enc = Encoder(config['vocab_size'], config['d_word_dim'], config['n_layers'],
                                config['n_heads'], config['d_k'], config['d_v'], config['d_model'],
                                config['d_ffn'], config['pad_idx'])
        self.api_enc = Encoder(config['vocab_size'], config['d_word_dim'], config['n_layers'],
                               config['n_heads'], config['d_k'], config['d_v'], config['d_model'],
                               config['d_ffn'], config['pad_idx'])
        self.token_enc = Encoder(config['vocab_size'], config['d_word_dim'], config['n_layers'],
                                 config['n_heads'], config['d_k'], config['d_v'], config['d_model'],
                                 config['d_ffn'], config['pad_idx'])
        self.desc_enc = Encoder(config['vocab_size'], config['d_word_dim'], config['n_layers'],
                                config['n_heads'], config['d_k'], config['d_v'], config['d_model'],
                                config['d_ffn'], config['pad_idx'])
        self.joint_enc1 = JointEncoder(config['n_heads'], config['d_k'], config['d_v'], config['d_model'],
                                     config['d_ffn'])
        self.joint_enc2 = JointEncoder(config['n_heads'], config['d_k'], config['d_v'], config['d_model'],
                                       config['d_ffn'])

        self.fc_code = nn.Linear(config["d_model"], config["d_model"])
        self.fc_desc = nn.Linear(config["d_model"], config["d_model"])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def code_encoding(self, name, api, token):
        name_repr = self.name_enc(name)
        api_repr = self.api_enc(api)
        token_repr = self.token_enc(token)
        code_repr = torch.cat((name_repr, api_repr, token_repr), dim=1)

        return code_repr

    def description_encoding(self, desc):
        desc_repr = self.desc_enc(desc)

        return desc_repr

    def joint_encoding(self, repr1, repr2, repr3):
        batch_size = repr1.size(0)
        code_repr = self.joint_enc1(repr1, repr2)
        desc_pos_repr = self.joint_enc2(repr2, repr1)
        desc_neg_repr = self.joint_enc2(repr3, repr1)
        code_repr = torch.mean(self.fc_code(code_repr), dim=1)
        desc_pos_repr = torch.mean(self.fc_desc(desc_pos_repr), dim=1)
        desc_neg_repr = torch.mean(self.fc_desc(desc_neg_repr), dim=1)
        return code_repr, desc_pos_repr, desc_neg_repr

    def cal_similarity(self, code, desc):
        assert self.conf['sim_measure'] in \
               ['cos', 'poly', 'euc', 'sigmoid', 'gesd', 'aesd'], "invalid similarity measure"

        if self.conf["sim_measure"] == "cos":
            return F.cosine_similarity(code, desc)
        elif self.conf["sim_measure"] == "poly":
            return (0.5 * torch.matmul(code, desc.t()).diag() + 1) ** 2
        elif self.conf['sim_measure'] == 'sigmoid':
            return torch.tanh(torch.matmul(code, desc.t()).diag() + 1)
        elif self.conf['sim_measure'] in ['euc', 'gesd', 'aesd']:
            euc_dist = torch.dist(code, desc, 2)  # or torch.norm(code_vec-desc_vec,2)
            euc_sim = 1 / (1 + euc_dist)
            if self.conf['sim_measure'] == 'euc': return euc_sim
            sigmoid_sim = torch.sigmoid(torch.matmul(code, desc.t()).diag() + 1)
            if self.conf['sim_measure'] == 'gesd':
                return euc_sim * sigmoid_sim
            elif self.conf['sim_measure'] == 'aesd':
                return 0.5 * (euc_sim + sigmoid_sim)

    def forward(self, name, api, token, desc_pos, desc_neg):
        # Obtain the representations of code snippets and code description
        code_repr = self.code_encoding(name, api, token)
        desc_pos_repr = self.description_encoding(desc_pos)
        desc_neg_repr = self.description_encoding(desc_neg)
        code_repr, desc_pos_repr, desc_neg_repr = self.joint_encoding(code_repr, desc_pos_repr,
                                                                      desc_neg_repr)

        pos_sim = self.cal_similarity(code_repr, desc_pos_repr)
        neg_sim = self.cal_similarity(code_repr, desc_neg_repr)

        loss = (self.margin-pos_sim+neg_sim).clamp(min=1e-6).mean()

        return loss





