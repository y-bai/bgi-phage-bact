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
from joblib import dump
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler


class Preprocessor:
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

    def get_datadfs(self, recal=True):
        """
        build x_file and corresponding y_file(containing phenotype data like height etc)

        Parameters
        ----------
        recal: Boolean
            recalculate data sets of train, validation and test
            if recal = False, then function loads csv files
        Returns
        -------
        trn_set_df: dict
        val_set_df: dict
        tst_set_df: dict

        the corresponding data frames are also saved in the path indicated by `data_dir` of `config.json`.
        """

        # those files contain score >1
        trn_fname = os.path.join(self.save_dir, 'reg_trn_set_df.csv')
        val_fname = os.path.join(self.save_dir, 'reg_val_set_df.csv')
        tst_fname = os.path.join(self.save_dir, 'reg_tst_set_df.csv')

        if os.path.exists(trn_fname) and os.path.exists(val_fname) and os.path.exists(tst_fname) and not recal:
            logging.info('reading...\n{}\n{}\n{}'.format(trn_fname, val_fname, tst_fname))
            trn_set_df = pd.read_csv(trn_fname, sep='\t')
            val_set_df = pd.read_csv(val_fname, sep='\t')
            tst_set_df = pd.read_csv(tst_fname, sep='\t')
        else:
            # read phage and bact relation file
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
            data_info_sel = data_info_df[data_info_df['b_Score'] > 1].copy()
            data_info_sel.reset_index(drop=True, inplace=True)

            b_scores = data_info_sel['b_Score'].values
            b_scores = b_scores.reshape(-1, 1)

            # MinMaxScaler()
            mm_scaler = MinMaxScaler()
            mm_scaler.fit(b_scores)
            mm_scaler_fname = os.path.join(self.save_dir, 'minmax_scaler.joblib')
            if os.path.exists(mm_scaler_fname):
                os.remove(mm_scaler_fname)
            dump(mm_scaler, mm_scaler_fname)
            data_info_sel['minmax_bscore'] = np.squeeze(mm_scaler.transform(b_scores))

            # StandardScaler()
            ss_scaler = StandardScaler()
            ss_scaler.fit(b_scores)
            ss_scaler_fname = os.path.join(self.save_dir, 'standard_scaler.joblib')
            if os.path.exists(ss_scaler_fname):
                os.remove(ss_scaler_fname)
            dump(ss_scaler, ss_scaler_fname)
            data_info_sel['norm_bscore'] = np.squeeze(ss_scaler.transform(b_scores))

            # MaxAbsScaler()
            ma_scaler = MaxAbsScaler()
            ma_scaler.fit(b_scores)
            ma_scaler_fname = os.path.join(self.save_dir, 'maxabs_scaler.joblib')
            if os.path.exists(ma_scaler_fname):
                os.remove(ma_scaler_fname)
            dump(ma_scaler, ma_scaler_fname)
            data_info_sel['maxabs_bscore'] = np.squeeze(ma_scaler.transform(b_scores))

            logging.info('number of samples with score > 1 : {}'.format(len(data_info_sel)))

            # creat train, validation and test dataset
            # split dataframe, 80% for train , 10% for validation and 10% for test
            trn_set_df, val_set_df, tst_set_df = np.split(data_info_sel.sample(frac=1, random_state=1),
                                                          [int(.7*len(data_info_sel)), int(.85*len(data_info_sel))])

            trn_set_df.reset_index(drop=True, inplace=True)
            val_set_df.reset_index(drop=True, inplace=True)
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

        return trn_set_df, val_set_df, tst_set_df









