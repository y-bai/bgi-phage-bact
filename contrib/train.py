#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: train.py
    Description:
    
Created by YongBai on 2020/8/19 10:55 AM.
"""
import argparse
import logging
import numpy as np
import pandas as pd
from logger import get_logger
from prepare import *
from data import PhageBactDataLoader
from model import *

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras import backend as K

# fix random seeds for reproducibility
SEED = 123
# torch.manual_seed(SEED)
# # accelerate pytorch
# # using deterministic cudnn convolutional opt(like seed = 0 so that all results of cnn can be reproducible)
# torch.backends.cudnn.deterministic = True
# # if input and model on the computation graph not changing,
# # then benchmark = True would help speed up pytorch
# torch.backends.cudnn.benchmark = True
tf.random.set_seed(SEED)


def get_trn_val_data(config, proc_data_type='cls', recal=False):
    # read data files
    if proc_data_type == 'cls':
        prep = ClsPreprocessor(config)
    else:
        prep = Preprocessor(config)
    trn_set_df, val_set_df, tst_set_df, class_weight_dict = prep.get_datadfs(recal=recal)
    data_root_dir = '{}/encode_data'.format(config['data_dir'])
    batch_size = config['hparams']['batch_size']

    batch_trn_data = PhageBactDataLoader(trn_set_df, batch_size, data_root_dir)
    batch_val_data = PhageBactDataLoader(val_set_df, batch_size, data_root_dir, shuffle=False)
    # batch_tst_data = PhageBactDataLoader(tst_set_df, batch_size, data_root_dir, shuffle=False)

    return len(trn_set_df), len(val_set_df), batch_trn_data, batch_val_data, class_weight_dict


def set_callbacks(config, monitor_metric='val_auc', mode_spe='max'):

    save_dir = config['resparams']['save_model_dir']
    runid = config['resparams']['run_id']

    csv_logger = CSVLogger(os.path.join(save_dir, '{}-csvlogger.log'.format(runid)))
    early_stop = EarlyStopping(monitor=monitor_metric,
                               min_delta=0.00001,
                               patience=10,
                               verbose=1,
                               mode=mode_spe)
    model_chkpt = ModelCheckpoint(filepath=os.path.join(save_dir, '{}-model.h5'.format(runid)),
                                  save_weights_only=True,
                                  save_best_only=True,
                                  monitor=monitor_metric,
                                  verbose=1,
                                  mode=mode_spe)

    lr_reduce_plt = ReduceLROnPlateau(monitor=monitor_metric,
                                      factor=0.1,
                                      patience=5,
                                      mode=mode_spe,
                                      min_delta=0.000000001,
                                      cooldown=0,
                                      verbose=1,
                                      min_lr=0)

    # tensorboard = TensorBoard(log_dir=os.path.join(save_dir, '{}-tensorboard.log'.format(runid)),
    #                           histogram_freq=1)
    return list([csv_logger, early_stop, model_chkpt, lr_reduce_plt])


def train_run(config, whole_len_trn, whole_len_val, batch_trn_data, batch_val_data, class_weight_dict):
    batch_size = config['hparams']['batch_size']
    init_lr = config['hparams']['lr']
    l2_weight = config['hparams']['l2_weight_decay']

    multi_gpu = config['trnparams']['multi_gpu']
    epochs = config['trnparams']['epochs']
    n_cpus = config['trnparams']['n_cups']

    # multi-GPU
    if multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            # initialize model
            # model = phage_bact_net_baseline(5, l2_weight)
            model = phage_bact_net_bilstm(5, l2_weight)
            # optimizer = tf.keras.optimizers.RMSprop(0.001)
            optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)
            # model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[
                tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC()])

    else:

        # initialize model
        # model = phage_bact_net_baseline(5, l2_weight)
        model = phage_bact_net_bilstm(5, l2_weight)
        # optimizer = tf.keras.optimizers.RMSprop(0.001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()])

    # This `fit` call will be distributed on 4 GPUs.
    # Since the batch size is 32, each GPU will process 8 samples.
    model.fit(x=batch_trn_data,
              steps_per_epoch=(whole_len_trn // batch_size),
              epochs=epochs,
              verbose=1,
              class_weight=class_weight_dict,
              callbacks=set_callbacks(config),
              validation_data=batch_val_data,
              validation_steps=(whole_len_val // batch_size))


def pred_run(config):

    l2_weight = config['hparams']['l2_weight_decay']
    runid = config['resparams']['run_id']

    model_weight_path = os.path.join(config['resparams']['save_model_dir'], '{}-model.h5'.format(runid))

    model = phage_bact_net_bilstm(5, l2_weight)
    model.load_weights(model_weight_path)

    # load test data
    prep = Preprocessor(config)
    _, _, tst_set_df = prep.get_datadfs(recal=False)
    logging.info('test sample size = {}'.format(len(tst_set_df)))
    data_root_dir = '{}/encode_data'.format(config['data_dir'])

    y_pred = []
    y_true = []
    for row_idx, row in tst_set_df.iterrows():
        y_score = row[2]  # b_Score
        x_data_fname = row[3]  # phage_bac_name

        tmp = np.load(os.path.join(data_root_dir, '{}.npz'.format(x_data_fname)), allow_pickle=True)
        tmp_bact = tmp['bac_onehot'].tolist().toarray()
        tmp_bact = np.expand_dims(tmp_bact, axis=0)
        tmp_phage = tmp['phage_onehot'].tolist().toarray()
        tmp_phage = np.expand_dims(tmp_phage, axis=0)
        pred_y = model.predict([tmp_phage, tmp_bact])
        pred_y = np.squeeze(pred_y)
        logging.info('y_true: {:.4f}, y_pred:{:.4f}'.format(y_score, pred_y))
        y_pred.append(pred_y)
        y_true.append(y_score)

    pred_out_path = os.path.join(config['resparams']['save_model_dir'], '{}-pred-out.csv'.format(runid))
    if os.path.exists(pred_out_path):
        os.remove(pred_out_path)
    pred_df = pd.DataFrame(data={'y_pred': y_pred, 'y_true': y_true})
    pred_df.to_csv(pred_out_path, index=False)

    logging.info('prediction DONE')


def findlr_run(config, whole_len_trn, whole_len_val, batch_trn_data, batch_val_data):

    import matplotlib
    import matplotlib.pyplot as plt

    batch_size = config['hparams']['batch_size']
    init_lr = config['hparams']['lr']
    l2_weight = config['hparams']['l2_weight_decay']

    multi_gpu = config['trnparams']['multi_gpu']
    epochs = 3
    n_cpus = config['trnparams']['n_cups']
    # multi-GPU
    if multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            # initialize model
            # model = phage_bact_net_baseline(5, l2_weight)
            model = phage_bact_net_bilstm(5, l2_weight)
            # optimizer = tf.keras.optimizers.RMSprop(0.001)
            # optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)
            model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    else:
        # initialize model
        # model = phage_bact_net_baseline(5, l2_weight)
        model = phage_bact_net_bilstm(5, l2_weight)
        # optimizer = tf.keras.optimizers.RMSprop(0.001)
        # optimizer = tf.keras.optimizers.Adam()
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    # This `fit` call will be distributed on 4 GPUs.
    # Since the batch size is 32, each GPU will process 8 samples.
    nb_steps = whole_len_trn // batch_size

    lr_finder = LRFinder(nb_steps=nb_steps)
    model.fit(x=batch_trn_data,
              steps_per_epoch=nb_steps,
              epochs=epochs,
              verbose=1,
              # class_weight=None,  # class_weight=class_weights
              callbacks=[lr_finder])

    lrs, losses = (lr_finder.history["lr"], lr_finder.history["batch_loss"])
    _, ax = plt.subplots(1, 1)
    ax.plot(lrs, losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    # ax.set_xscale('log10')
    # ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))

    plt.savefig(os.path.join(config['resparams']['save_model_dir'], "{}-lr.pdf".format(config['resparams']['run_id'])),
                bbox_inches='tight', pad_inches=0.1)


def main(in_args):

    # setup paramters
    paranm_config = Config.init_params(in_args.config).config
    # get logger
    logger = get_logger(__name__)
    logger.debug(config)

    logger.info('tensorflow veriosn: {}'.format(tf.__version__))
    K.clear_session()

    ######################################
    # train model
    # get train data and validation data
    whole_len_trn, whole_len_val, batch_trn_data, batch_val_data, class_weight_dict = get_trn_val_data(
        paranm_config, recal=False)
    # get_trn_val_data(paranm_config)
    #
    # # findlr_run(paranm_config, whole_len_trn, whole_len_val, batch_trn_data, batch_val_data)
    train_run(paranm_config, whole_len_trn, whole_len_val, batch_trn_data, batch_val_data, class_weight_dict)

    #####################################
    # test model
    # run_id = '1009_214731'
    # paranm_config['resparams']['run_id'] = run_id
    # pred_run(paranm_config)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Phage-bact NET')
    args.add_argument('-c', '--config',
                      default=None, type=str,
                      help='config json file path')
    args = args.parse_args()
    main(args)
