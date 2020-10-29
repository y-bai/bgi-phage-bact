#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: preprocess.py
    Description:
    
Created by YongBai on 2020/8/19 5:43 PM.
"""

import logging
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


class ClsPreprocessor:
    def __init__(self, config):
        """
        Preprocessor to get training, validation and testing files.

        Parameters
        ----------
        config: dict

        """
        self.config = config

        self.data_dir = self.config['data_dir']
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError('data directory not found {}.'.format(self.data_dir))
        self.save_dir = self.config['resparams']['save_dir']
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def get_datadfs(self, recal=True, nb_neg_to_pos=2):
        """
        build x_file and corresponding y_file(containing phenotype data like height etc)

        Parameters
        ----------
        recal: Boolean
            recalculate data sets of train, validation and test
            if recal = False, then function loads csv files

        nb_neg_to_pos: the ratio of the number of negative samples to the number of positive samples
            if None, then using all samples
        Returns
        -------
        trn_set_df: dict
        val_set_df: dict
        tst_set_df: dict

        the corresponding data frames are also saved in the path indicated by `data_dir` of `config.json`.
        """

        # those files contain score >1
        trn_fname = os.path.join(self.save_dir, 'reg_trn_set_df_cls.csv')
        val_fname = os.path.join(self.save_dir, 'reg_val_set_df_cls.csv')
        tst_fname = os.path.join(self.save_dir, 'reg_tst_set_df_cls.csv')

        if os.path.exists(trn_fname) and os.path.exists(val_fname) and os.path.exists(tst_fname) and not recal:
            logging.info('reading...\n{}\n{}\n{}'.format(trn_fname, val_fname, tst_fname))
            trn_set_df = pd.read_csv(trn_fname, sep='\t')
            val_set_df = pd.read_csv(val_fname, sep='\t')
            tst_set_df = pd.read_csv(tst_fname, sep='\t')
        else:
            # read phage and bact relation file
            random_seed = 1
            data_meta_info_f = self.config['data_meta_info']
            if not os.path.exists(data_meta_info_f):
                logging.error('data meta information file not Found. {}'.format(data_meta_info_f))
                raise FileNotFoundError('data meta information file not Found')
            data_info = pd.read_csv(data_meta_info_f)
            logging.info('total samples: {}'.format(len(data_info)))

            # select specific columns
            sel_cols = 'hostID,phageID,b_Score,phage_bac_name,bac_len,phage_len'.split(',')
            data_info_df = data_info[sel_cols]

            # select rows with score > 1
            data_info_sel_pos = data_info_df[data_info_df['b_Score'] >= 1].copy()
            data_info_sel_pos.reset_index(drop=True, inplace=True)
            nb_pos = len(data_info_sel_pos)
            logging.info('the number of samples with score >= 1 (positive): {}'.format(nb_pos))
            data_info_sel_pos['label'] = [1] * nb_pos
            trn_set_df_pos, val_set_df_pos, tst_set_df_pos = np.split(
                data_info_sel_pos.sample(frac=1, random_state=random_seed),
                [int(.9 * len(data_info_sel_pos)), int(.95 * len(data_info_sel_pos))])

            data_info_sel_neg_tmp = data_info_df[data_info_df['b_Score'] == 0].copy()
            data_info_sel_neg_tmp.reset_index(drop=True, inplace=True)
            if nb_neg_to_pos is not None:
                data_info_sel_neg = data_info_sel_neg_tmp.sample(n=nb_pos * nb_neg_to_pos, random_state=random_seed)
            else:
                data_info_sel_neg = data_info_sel_neg_tmp
            data_info_sel_neg.reset_index(drop=True, inplace=True)
            nb_neg = len(data_info_sel_neg)
            logging.info('the number of samples with score < 1 (negative): {}'.format(nb_neg))
            data_info_sel_neg['label'] = [0] * nb_neg

            trn_set_df_neg, val_set_df_neg, tst_set_df_neg = np.split(
                data_info_sel_neg.sample(frac=1, random_state=random_seed),
                [int(.9 * len(data_info_sel_neg)), int(.95 * len(data_info_sel_neg))])

            trn_set_df = pd.concat([trn_set_df_pos, trn_set_df_neg])
            trn_set_df = trn_set_df.sample(frac=1.0, random_state=random_seed)
            trn_set_df.reset_index(drop=True, inplace=True)

            val_set_df = pd.concat([val_set_df_pos, val_set_df_neg])
            val_set_df = val_set_df.sample(frac=1.0, random_state=random_seed)
            val_set_df.reset_index(drop=True, inplace=True)

            tst_set_df = pd.concat([tst_set_df_pos, tst_set_df_neg])
            tst_set_df = tst_set_df.sample(frac=1.0, random_state=random_seed)
            tst_set_df.reset_index(drop=True, inplace=True)

            if os.path.exists(trn_fname):
                os.remove(trn_fname)
            trn_set_df.to_csv(trn_fname, sep='\t', index=False)
            if os.path.exists(val_fname):
                os.remove(val_fname)
            val_set_df.to_csv(val_fname, sep='\t', index=False)
            if os.path.exists(tst_fname):
                os.remove(tst_fname)
            tst_set_df.to_csv(tst_fname, sep='\t', index=False)

        logging.info('train sample: {}'.format(len(trn_set_df)))
        logging.info('valid sample: {}'.format(len(val_set_df)))
        logging.info('test sample: {}'.format(len(tst_set_df)))

        class_weight_dict = self.__get_class_weight(trn_set_df['label'].values)
        logging.info('classes weight: {}'.format(class_weight_dict))

        return trn_set_df, val_set_df, tst_set_df, class_weight_dict

    def __get_class_weight(self, y_train):
        labels = np.unique(y_train)
        cls_weight = compute_class_weight('balanced', labels, y_train)
        class_weight_dict = dict(zip(labels, cls_weight))
        return class_weight_dict








