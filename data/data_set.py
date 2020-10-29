#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: data_set.py
    Description:
    
Created by YongBai on 2020/10/14 1:48 PM.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import skimage
import logging


class PRSDataset(Dataset):
    def __init__(self, in_df, data_root_dir, embed_id=0):
        """

        Parameters
        ----------
        in_df
        data_root_dir
        embed_id: 0: one-hot, 1: 3-mer fixed sliding win length, 2: 4-mer fixed sliding win lenth.
        """
        self.in_df = in_df  # train_df
        self.data_root_dir = data_root_dir
        self.embed_id = embed_id

    def __len__(self):
        return len(self.in_df)

    def __getitem__(self, item: int):

        phage_id = self.in_df.iloc[item, 1]
        bact_id = self.in_df.iloc[item, 0]
        x_fname = self.in_df.iloc[item, 3]

        y = self.in_df.iloc[item, 6]  # label
        x = self.__get_data(x_fname, phage_id, bact_id)

        # normalization
        # https://nlp.stanford.edu/IR-book/html/htmledition/maximum-tf-normalization-1.html
        x_max = np.amax(x, axis=0)
        x_max[x_max == 0] = 0.00001
        # x_max = np.expand_dims(x_max, axis=0)
        # print(x_max)
        # print(x_max.shape)

        alpha = 0.4
        norm_x = alpha + (1.0 - alpha) * x / x_max

        return norm_x, y.astype(np.float32)

    def __get_data(self, fname, phage_id=None, bact_id=None):

        if self.embed_id == 0:
            # read one hot encoded data with original DNA length
            tmp = np.load(os.path.join(self.data_root_dir, '{}.npz'.format(fname)), allow_pickle=True)
            tmp_bact = np.transpose(tmp['bac_onehot'].tolist().toarray())  # tmp_bact.size = [5, n]
            tmp_phage = np.transpose(tmp['phage_onehot'].tolist().toarray())
            return tmp_phage.astype(np.float32), tmp_bact.astype(np.float32)

        else:
            # reading 3-mer/4-mer fixed sliding win length
            SEQ_LEN = 11000
            phage_fname = os.path.join(self.data_root_dir, 'phage_500/{}.countMatrix.csv'.format(phage_id))
            bact_fname = os.path.join(self.data_root_dir, 'bac_1000/{}.countMatrix.csv'.format(bact_id))

            phage_data = np.transpose(pd.read_csv(phage_fname, header=None).values.astype(np.float32))
            bact_data = np.transpose(pd.read_csv(bact_fname, header=None).values.astype(np.float32))
            x_phage_bact = np.concatenate((phage_data, bact_data), axis=-1)

            batch_x = np.zeros((x_phage_bact.shape[0], SEQ_LEN), dtype=np.float32)

            if x_phage_bact.shape[-1] > SEQ_LEN:
                batch_x = x_phage_bact[:, :SEQ_LEN]
            else:
                batch_x[:, :x_phage_bact.shape[-1]] = x_phage_bact

            return batch_x


def get_slide(seq_array, fix_len):

    nb_feats, total_len = seq_array.shape

    if total_len < fix_len:
        return seq_array

    # logging.info('sliding window...')
    # np.floor((L-win)/stride + 1) = phage_fix_len
    # first let win = 2 * stride
    strid = int(np.floor(total_len / (fix_len + 1)))
    win = total_len - (fix_len - 1) * strid

    slide_array = skimage.util.view_as_windows(seq_array,
                                               (nb_feats, win),
                                               step=strid).squeeze().sum(axis=-1)
    slide_array_max = np.expand_dims(slide_array.max(axis=-1), axis=1)

    # normalization
    # https://nlp.stanford.edu/IR-book/html/htmledition/maximum-tf-normalization-1.html
    alpha = 0.4
    re_slide = alpha + (1.0 - alpha) * slide_array / slide_array_max

    return np.transpose(re_slide)







