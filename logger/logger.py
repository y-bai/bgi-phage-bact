#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: logger.py
    Description:
    
Created by YongBai on 2020/8/19 2:27 PM.
"""

import logging
import logging.config
from pathlib import Path
from utils import read_json


def setup_logging(save_dir, run_id='1', log_config='logger/logger.json', default_level=logging.INFO):
    """
    Setup logging configuration

    Parameters
    ----------
    save_dir: str
        logging info save root directory. By default, the name of logging file is info.log.
    run_id: str
        session id.
    log_config: str, default='logger/logger.json'
        logging config json file.
    default_level: str, default=logging.INFO
        logging level.

    Returns
    -------
    None
    """

    _log_config = Path(log_config)
    if _log_config.is_file():
        # read json file
        config_dict = read_json(_log_config)
        for _, handler in config_dict['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir/'{}-{}'.format(run_id, handler['filename']))
        logging.config.dictConfig(config_dict)

    else:
        print("Warning: logging configuration file is not found in {}.".format(_log_config))
        logging.basicConfig(level=default_level)


def get_logger(name, verbosity=2):
    """

    Parameters
    ----------
    name
    verbosity

    Returns
    -------

    """

    log_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                   log_levels.keys())
    assert verbosity in log_levels, msg_verbosity
    logger = logging.getLogger(name)
    logger.setLevel(log_levels[verbosity])
    return logger

