#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: train.py
    Description:
    
Created by YongBai on 2020/10/14 2:21 PM.
"""

import os
import argparse
import torch
import torch.nn as nn
from logger import get_logger
from prepare import *
from data import PRSDataset
from model import *
from torchsummary import summary

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)  # CPU seed
torch.cuda.manual_seed(SEED)  # GPU seed
# accelerate pytorch
# using deterministic cudnn convolutional opt(like seed = 0 so that all results of cnn can be reproducible)
torch.backends.cudnn.deterministic = True
# if input and model on the computation graph not changing,
# then benchmark = True would help speed up pytorch
torch.backends.cudnn.benchmark = True


def main(in_args):

    embed_id = 1   # 0: one-hot, 1: 3-mer fixed length sliding window,, 2: 4-mer fixed length sliding window
    nb_feats = 5 ** 3 if embed_id == 1 else 5 ** 4

    # setup paramters
    config = Config.init_params(in_args.config).config

    # get logger
    logger = get_logger(__name__)
    logger.debug(config)

    logger.info('Pytorch version: {}'.format(torch.__version__))
    logger.info('cuda version:{}'.format(torch.version.cuda))

    # read data files
    prep = ClsPreprocessor(config)
    data_root_dir = '{}/encode_data'.format(config['data_dir']) if embed_id == 0 \
        else config['data_fixwinlen_dirk3'] if embed_id == 1 else config['data_fixwinlen_dirk4']
    trn_set_df, val_set_df, tst_set_df, class_weight_dict = prep.get_datadfs(recal=True, nb_neg_to_pos=2)

    trn_dataset = PRSDataset(trn_set_df, data_root_dir, embed_id=embed_id)
    val_dataset = PRSDataset(val_set_df, data_root_dir, embed_id=embed_id)
    tst_dataset = PRSDataset(tst_set_df, data_root_dir, embed_id=embed_id)

    training_generator = torch.utils.data.DataLoader(trn_dataset,
                                                     batch_size=config['hparams']['batch_size'],
                                                     shuffle=True,
                                                     # collate_fn=coll_fn1,
                                                     num_workers=config['trnparams']['n_cups'])
    val_generator = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=config['hparams']['batch_size'],
                                                shuffle=False,
                                                # collate_fn=coll_fn1,
                                                num_workers=config['trnparams']['n_cups'])
    tst_generator = torch.utils.data.DataLoader(tst_dataset,
                                                batch_size=config['hparams']['batch_size'],
                                                shuffle=False,
                                                # collate_fn=coll_fn1,
                                                num_workers=config['trnparams']['n_cups'])
    #
    # batch_data_phage, batch_data_bact, y = next(iter(training_generator))
    # logger.info('batch_data_phage shape: {}'.format(batch_data_phage.shape))  #
    # logger.info(' batch_data_bact shape: {}, '.format(batch_data_bact.shape))  #
    # logger.info(' label shape: {}, '.format(y.shape))  #
    # logger.info(' labels: {}, '.format(y))  #

    # for vals in val_generator:
    #     i_data, i_y = vals
    #     print(i_data.shape)
    #     print(i_y)
    # print(i_data)

    logger.info('train data set length: {}'.format(len(training_generator)))
    logger.info('valid data set length: {}'.format(len(val_generator)))
    logger.info('test data set length: {}'.format(len(tst_generator)))
    #
    save_model_dir = config['resparams']['save_model_dir']
    run_id = config['resparams']['run_id']
    # run_id = '1012_161019'
    save_best_model_fname = os.path.join(save_model_dir, '{}_best_model.ckpt'.format(run_id))
    csv_metrics_fname = os.path.join(save_model_dir, '{}_train_metrics.csv'.format(run_id))
    pred_out_fname = os.path.join(save_model_dir, '{}_pred_out.csv'.format(run_id))
    ############################
    # train AE model
    # model = PhageBactAutoEncoder()
    # logger.info(model)
    # logger.info(summary(model=model.cuda(), input_size=(125, 11000), batch_size=config['hparams']['batch_size']))
    # gpus_b = config['trnparams']['multi_gpu']
    # gpus = [0, 1, 2, 3] if gpus_b else []
    #
    # trainer = Trainer(save_best_model_fname, csv_metrics_fname,
    #                   nb_epochs=config['trnparams']['epochs'],
    #                   lr=config['hparams']['lr'],
    #                   gpus=gpus, class_wieght=class_weight_dict)
    # trainer.train(model, training_generator, val_generator)
    # logger.info('finish training')

    ############################
    # train classifier
    # model = AECls()
    # model = AEConvClsa()
    model = AESECls()
    logger.info(model)
    logger.info(summary(model=model.cuda(), input_size=(125, 11000), batch_size=config['hparams']['batch_size']))

    # loaded pre-trained AE part
    f_model_dir = config['resparams']['save_final_dir']
    pretrained_ae_path = os.path.join(f_model_dir, '1027_221931_best_ae_model.ckpt')
    chechpoint = torch.load(pretrained_ae_path)
    model.load_state_dict(chechpoint, strict=False)
    count = 0
    for param in model.encoder.parameters():
        count += 1
        if count < 10:
            param.requires_grad = False

    logger.info('number of layers in model.encoder: {}'.format(count))  # count=24
    gpus_b = config['trnparams']['multi_gpu']
    gpus = [0, 1, 2, 3] if gpus_b else []

    trainer = Trainer(save_best_model_fname, csv_metrics_fname,
                      nb_epochs=config['trnparams']['epochs'],
                      lr=config['hparams']['lr'],
                      gpus=gpus, class_wieght=class_weight_dict)
    trainer.train(model, training_generator, val_generator)
    logger.info('finish training')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PhageBactNET')
    args.add_argument('-c', '--config',
                      default=None, type=str,
                      help='config json file path')
    args = args.parse_args()
    main(args)