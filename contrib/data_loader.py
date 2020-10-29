#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: data_loader.py
    Description:
    2
Created by YongBai on 2020/8/21 4:30 PM.
"""

import os
import numpy as np
import math

from tensorflow.keras.utils import Sequence


def _get_data(fname):
    # in_data = np.asarray(scipy.sparse.load_npz(fname).todense(), dtype=np.float32)
    in_data = np.load(fname, dtype=np.float32)
    return in_data


class PhageBactDataLoader(Sequence):

    def __init__(self, in_df, batch_size, data_root_dir, nb_feats=5, shuffle=True):
        """

        Parameters
        ----------
        in_df: input dataframe with path of data
            header: hostID,phageID,b_Score,phage_bac_name,bac_len,phage_len
        batch_size
        shuffle
        """
        self.in_df = in_df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_len = len(self.in_df)
        self.data_root_dir = data_root_dir
        self.nb_feats = nb_feats

        self.on_epoch_end()

    def __len__(self):
        """
        the number of batches
        Returns
        -------

        """
        return math.ceil(self.sample_len / self.batch_size)

    def __getitem__(self, idx: int):

        # get batch data
        # get x-input file name
        # avoid possible deadlock when reach epoch end
        # ref: https://www.kaggle.com/ezietsman/keras-convnet-with-fit-generator
        # with self.lock:
        end_slice = min((idx + 1) * self.batch_size, self.sample_len)
        sel_idx = self.sample_idx_arr[idx * self.batch_size:end_slice]

        return self.__gen_batch_data(sel_idx)

    def __gen_batch_data(self, sample_batch_idx):

        batch_scores = self.in_df.iloc[sample_batch_idx, 6].values  # label
        batch_data_fnames = self.in_df.iloc[sample_batch_idx, 3].values  #

        batch_bact_len = self.in_df.iloc[sample_batch_idx, 4].values
        batch_phage_len = self.in_df.iloc[sample_batch_idx, 5].values

        bact_max_len = max(batch_bact_len)
        phage_max_len = max(batch_phage_len)

        batch_data_phage = np.zeros((len(sample_batch_idx), phage_max_len, self.nb_feats), dtype='float32')
        batch_data_bact = np.zeros((len(sample_batch_idx), bact_max_len, self.nb_feats), dtype='float32')

        for batch_idx, fname in enumerate(batch_data_fnames):
            tmp = np.load(os.path.join(self.data_root_dir, '{}.npz'.format(fname)), allow_pickle=True)
            tmp_bact = tmp['bac_onehot'].tolist().toarray()
            tmp_phage = tmp['phage_onehot'].tolist().toarray()

            batch_data_bact[batch_idx, :tmp_bact.shape[0], :tmp_bact.shape[1]] = tmp_bact
            batch_data_phage[batch_idx, :tmp_phage.shape[0], :tmp_phage.shape[1]] = tmp_phage

        return [batch_data_phage, batch_data_bact], batch_scores

    def on_epoch_end(self):
        """
        Updates indexes after each epoch

        :return:
        """
        self.sample_idx_arr = np.arange(self.sample_len)

        if self.shuffle:
            np.random.shuffle(self.sample_idx_arr)
