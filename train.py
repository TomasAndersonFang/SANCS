#!/usr/bin/env python
# encoding: utf-8
"""
@author: Tomas S. Fang
@contact: fangsen1996@gmail.com
@software: PyCharm
@file: train.py
@time: 2020/9/18 19:39
"""


import torch
import torch.optim as optim
import os
import time
import sys
import numpy as np
import argparse
import configs
from loguru import logger
from tqdm import tqdm
from datetime import datetime
from dataset import Dataset, load_dict
import method.jointemb as jointemb
from utils import validate


#logger.add("./data/train.log")


def train_epoch(model, train_data, validation_data, optimizer, config, args,  device):
    """Epoch operation in the training pharse."""

    model.train()
    losses = []
    itr_start_time = time.time()
    n_itr = len(train_data)
    text = "---------- Training ----------"
    def save_model(model, path):
        torch.save(model.state_dict(), path)

    #for index, batch in enumerate(tqdm(train_data, mininterval=2, desc=text, leave=False)):
    for index, batch in enumerate(train_data):
        batch = [data.to(device) for data in batch][0::2]
        optimizer.zero_grad()
        loss = model(*batch)

        if config['fp16']:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        losses.append(loss.item())

        if (index+1) % args.log_every == 0:
            elapsed = time.time() - itr_start_time
            info = 'itr:{} step_time:{} Loss={}'.format((index+1), elapsed/(index+1), np.mean(losses))
            logger.info(info)

        if (index+1) % args.valid_every == 0:
            logger.info('Validating.')
            re, acc, mrr, map, ndcg = validate(validation_data, model, 1, 'cos')
            result = 'Recall={}, Accurate={}, Mrr={}, Map={}, NDCG={}'.format(re, acc, mrr, map, ndcg)
            print(result)

    return losses


def train(model, training_data, validation_data, optimizer, args, device):
    """Training."""
    logger.info("Start training!")
    best_mrr = 0.
    config = getattr(configs, 'config')()

    def save_model(model, path):
        torch.save(model.state_dict(), path)

    for epoch in range(args.Epoch):
        info = '[ Epoch ' + str(epoch) + ' ]'
        logger.info(info)
        train_loss = train_epoch(model, training_data, validation_data, optimizer, config, args, device)
        logger.info("The loss of epoch {} is: {}".format(epoch, np.mean(train_loss)))

        logger.info("Validating.")
        re, acc, mrr, _, _ = validate(validation_data, model, 1, config['sim_measure'])

        valid_mrr = mrr
        if best_mrr < valid_mrr:
            best_mrr = valid_mrr
            print("The current best mrr score is: ", best_mrr)
            path = args.model_path + 'joint_embed_model.h5'
            save_model(model, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='JointEmbedder', help='model name')
    parser.add_argument('--dataset', type=str, default='github/', help='name of dataset.java, python')
    parser.add_argument('--reload_from', type=int, default=-1, help='epoch to reload from')
    parser.add_argument('--model_path', type=str, default='./model_save/', help='path of saving model')
    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('-v', "--visual", action="store_true", default=False,
                        help="Visualize training status in tensorboard")
    parser.add_argument('--best_mrr', type=float, default=0., help='The MRR metric.')

    parser.add_argument('--log_every', type=int, default=1000, help='interval to log autoencoder training results')
    parser.add_argument('--valid_every', type=int, default=30000, help='interval to validation')
    parser.add_argument('--save_every', type=int, default=10000, help='interval to evaluation to concrete results')

    parser.add_argument('--sim_measure', type=str, default='cos', help='similarity measure for training')
    parser.add_argument('--Epoch', type=int, default=15, help="Training Epoch")

    args = parser.parse_args()

    config = getattr(configs, 'config')()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ****** Load Dataset ******
    data_path = args.data_path + args.dataset

    valid_set = Dataset(data_path, config['valid_name'], config['name_len'],
                        config['valid_api'], config['api_len'], config['valid_tokens'],
                        config['tokens_len'], config['valid_desc'], config['desc_len'])

    model = getattr(jointemb, args.model)(config)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], eps=config['adam_epsilon'])

    def load_model(model, ckpt_path, to_device):
        assert os.path.exists(ckpt_path), f'Weights not found'
        model.load_state_dict(torch.load(ckpt_path, map_location=to_device))

    def count_parameters(model):
       return sum((p.numel() for p in model.parameters() if p.requires_grad))

    print(f'The model has {count_parameters(model):,} trainable parameters!')

    if config["fp16"]:
        try:
            from apex import amp
        except:
            ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=config['fp16_opt_level'])

    if args.mode == 'train':
        train_set = Dataset(data_path, config['train_name'], config['name_len'],
                            config['train_api'], config['api_len'], config['train_tokens'],
                            config['tokens_len'], config['train_desc'], config['desc_len'])
        train_iter = torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'],
                                                 shuffle=True, drop_last=True, num_workers=1)
        model = model.to(device)
        train(model, train_iter, valid_set, optimizer, args, device)
    elif args.mode == 'eval':
        path = args.model_path + 'joint_embed_model.h5'
        load_model(model, path, device)
        model.to(device)
        start_time = time.time()
        K = [1, 5, 10]
        for k in K:
            re, acc, mrr, map, ndcg = validate(valid_set, model, k, 'cos')
            search_time = time.time() - start_time
            query_time = search_time / 10000
            results = "k={}, re={}, acc={}, mrr={}, ndcg={}, q_time={}".format(k, re, acc, mrr, ndcg, query_time)
            with open('result.txt', 'a') as f:
                f.write(results + '\n')
            print("The search time of each query is: {}".format(query_time))


if __name__ == "__main__":
    main()





