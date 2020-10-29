#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: config.py
    Description:
    
Created by YongBai on 2020/8/21 9:07 AM.
"""

import os
import logging
from pathlib import Path
from datetime import datetime

from utils.util import read_json
from logger.logger import setup_logging


class Config:
    def __init__(self, config_dict):
        self.config = config_dict

    @classmethod
    def init_params(cls, config_fname=None):
        """
        intialize and setup params from config file

        Parameters
        ----------
        config_fname

        Returns
        -------

        """
        if config_fname is None:
            config_fname = os.path.join(Path(__file__).parent.parent, 'config.json')
            logging.info('Using default configuration file: ', config_fname)

        # parse the config file
        _config = read_json(config_fname)
        # specify the saving directory
        proj_name = _config['project_name']
        save_root_dir = Path(_config['resparams']['save_dir'])
        run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        save_model_dir = save_root_dir / proj_name / 'models'
        save_log_dir = save_root_dir / proj_name / 'logs'

        if not os.path.exists(save_model_dir):
            save_model_dir.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(save_log_dir):
            save_log_dir.mkdir(parents=True, exist_ok=True)

        _config['resparams']['run_id'] = run_id
        _config['resparams']['save_model_dir'] = str(save_model_dir)
        _config['resparams']['save_log_dir'] = str(save_log_dir)

        # setup logging
        setup_logging(save_log_dir, run_id=run_id)

        return cls(_config)



