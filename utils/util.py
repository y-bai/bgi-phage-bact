#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: util.py
    Description:
    
Created by YongBai on 2020/8/19 12:51 PM.
"""
import os
import json
import numpy as np


def read_json(json_fn: str) -> dict:
    """
    read json  file

    Parameters
    ----------
    json_fn: str, default=None
        json file path.

    Returns
    -------
    dict
    """
    assert os.path.exists(json_fn), 'input json file not found: {}'.format(json_fn)
    with open(json_fn, 'r') as f:
        return json.load(f)


def read_npz(npz_fn, re_array=False):

    data_loads = np.load(npz_fn, allow_pickle=True)
    # npz_fn = './one-hot-raw/encode_data/PH-KP4547_5046.npz'
    # print(data_loads.files)
    # ['y_score', 'bac_onehot', 'phage_onehot', 'bac_file', 'phage_file']
    y_score = data_loads['y_score']
    if re_array:
        bac_one_hot_array = data_loads['bac_onehot'].tolist().toarray()
        phage_one_hot_array = data_loads['phage_onehot'].tolist().toarray()
        return y_score, bac_one_hot_array, phage_one_hot_array
    else:
        return y_score







