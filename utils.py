#!/usr/bin/env python
# encoding: utf-8
"""
@author: Tomas S. Fang
@contact: fangsen1996@gmail.com
@software: PyCharm
@file: utils.py
@time: 2020/9/15 17:01
"""
from abc import ABC

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import math


class PositionlEncoding(nn.Module, ABC):

    def __init__(self, d_hid, n_position=100):
        super(PositionlEncoding, self).__init__()

        self.register_buffer("pos_table", self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def normalize(data):
    """normalize matrix by rows"""
    return data / np.linalg.norm(data, axis=1, keepdims=True)


def validate(valid_set, model, k, sim_measure):
    """
    To evaluate the performance of our trained model.
    :param valid_set: Use it to evaluate trained model.
    :param model: Our trained model
    :param k: Take the top-k results
    :param sim_measure: The way to calculate similarity, in our experiments,
                        we use cosine similarity.
    :return:
    """

    def recall(gold, prediction, results):
        sum = 0.
        for val in gold:
            try:
                index = prediction.index(val)
            except ValueError:
                index = -1
            if index <= results:
                sum += 1
        return sum / float(len(gold))

    def acc(gold, prediction):
        sum = 0.
        for val in gold:
            try:
                index = prediction.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum += 1
        return sum / float(len(gold))

    def map(gold, prediction):
        sum = 0.
        for idx, val in enumerate(gold):
            try:
                index = prediction.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum = sum + (idx + 1) / float(index + 1)
        return sum / float(len(gold))

    def mrr(gold, prediction):
        sum = 0.
        for val in gold:
            try:
                index = prediction.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum = sum + 1.0 / float(index + 1)
        return sum / float(len(gold))

    def ndcg(gold, prediction):
        dcg = 0.
        idcgs = idcg(len(gold))
        for i, predictItem in enumerate(prediction):
            if predictItem in real:
                item_relevance = 1
                rank = i + 1
                dcg += (math.pow(2, item_relevance) - 1.0) * (math.log(2) / math.log(rank + 1))
        return dcg / float(idcgs)

    def idcg(n):
        idcg = 0
        item_relevance = 1
        for i in range(n):
            idcg += (math.pow(2, item_relevance) - 1.0) * (math.log(2) / math.log(i + 2))
        return idcg

    model.eval()
    # device = next(model.parameters()).device

    data_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=10000,
                                              shuffle=True, drop_last=True, num_workers=12)
    re, accu, mrrs, maps, ndcgs = 0., 0., 0., 0., 0.
    for batch in data_loader:
        if len(batch) == 10:  # names, name_len, apis, api_len, toks, tok_len, descs, desc_len, bad_descs, bad_desc_len
            code_batch = [tensor.cuda() for tensor in batch[:6]][0::2]
            desc_batch = [tensor.cuda() for tensor in batch[6:8]][0::2]
        with torch.no_grad():
            code_repr = model.code_encoding(*code_batch)
            desc_repr = model.description_encoding(*desc_batch)  # [batch_size, feature_dim]
            code_repr, desc_repr, _ = model.joint_encoding(code_repr, desc_repr, desc_repr)
            code_repr = code_repr.data.cpu()
            desc_repr = desc_repr.data.cpu()
            #if sim_measure == 'cos':
            #    code_repr = normalize(code_repr)
            #    desc_repr = normalize(desc_repr)

    data_len = code_repr.size(0)
    for i in tqdm(range(data_len), desc="-------- Eval --------"):  # for i in range(pool_size):
        desc_vec = desc_repr[i].unsqueeze(0)  # [1 x dim]
        n_results = k
        if sim_measure == 'cos':
            sims = torch.cosine_similarity(code_repr, desc_vec, dim=1)  # [pool_size]

        neg_sims = np.negative(sims)
        predict_origin = np.argsort(neg_sims)
        # predict = np.argpartition(negsims, kth=n_results - 1)
        predict = predict_origin[:n_results]
        predict = [int(k) for k in predict]
        predict_origin = [int(k) for k in predict_origin]
        real = [i]
        re += recall(real, predict_origin, n_results)
        accu += acc(real, predict)
        mrrs += mrr(real, predict)
        maps += map(real, predict)
        ndcgs += ndcg(real, predict)
    re = re / float(data_len)
    accu = accu / float(data_len)
    mrrs = mrrs / float(data_len)
    maps = maps / float(data_len)
    ndcgs = ndcgs / float(data_len)
    return re, accu, mrrs, maps, ndcgs
