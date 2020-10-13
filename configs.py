#!/usr/bin/env python
# encoding: utf-8
"""
@author: Tomas S. Fang
@contact: fangsen1996@gmail.com
@software: PyCharm
@file: configs.py
@time: 2020/8/11 10:36
"""


def config():

    conf = {
        'dataset_name': 'Dataset',  # name of dataset to specify a data loader
        # training data name
        'train_name': 'train.name.h5',
        'train_api': 'train.apiseq.h5',
        'train_tokens': 'train.tokens.h5',
        'train_desc': 'train.desc.h5',
        # valid data name
        'valid_name': 'test.name.h5',
        'valid_api': 'test.apiseq.h5',
        'valid_tokens': 'test.tokens.h5',
        'valid_desc': 'test.desc.h5',

        # data parameters
        'name_len': 6,
        'api_len': 30,
        'tokens_len': 50,
        'desc_len': 30,
        'vocab_size': 10000,

        # vocabulary info
        'vocab_name': 'vocab.name.json',
        'vocab_api': 'vocab.apiseq.json',
        'vocab_tokens': 'vocab.tokens.json',
        'vocab_desc': 'vocab.desc.json',

        # training parameters
        'batch_size': 256,
        'Epoch': 15,
        'learning_rate': 1e-4,
        'adam_epsilon': 1e-8,
        'warmup_steps': 5000,
        'fp16': False,

        # model parameters
        'd_word_dim': 128,
        'd_model': 128,
        'd_ffn': 512,
        'n_heads': 8,
        'n_layers': 1,
        'd_k': 16,
        'd_v': 16,
        'pad_idx': 0,
        'margin': 0.3986,
        'sim_measure': 'cos'
    }
    return conf
