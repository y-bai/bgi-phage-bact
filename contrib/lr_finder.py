#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: lr_finder.py
    Description:
    
Created by YongBai on 2020/10/10 10:07 AM.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K


class LRFinder(Callback):
    """
    Learning rate finder. This is a part of 1-cycle policy where we try to find the
    optimal learning rate for training a model.

    ref: https://colab.research.google.com/drive/1KG_A64xyZDP_TN5bMTEIuebBe7n3y3oQ#scrollTo=1DqrXaGDj4dg
    """

    def __init__(self,
                 base_lr=1e-7,
                 max_lr=1,
                 beta=0.98,
                 scale_min_loss=None,
                 nb_steps=None,
                 clip_initial_values=None,
                 clip_end_values=None):
        """
        Arguments:
            base_lr: Initial learning rate. Can be as low as 1e-8

            max_lr:  Maximum learning rate to be considered for the test. Although `high learning rate`
                     is a model dependent thing. For example, for a 3-layer network 0.1 can be a pretty high
                     learning rate. A value between 2-10 is fine and if you wish you can run the learning rate
                     test for more than one maximum value.

            beta:     Smoothening parameter for moving average of loss values. Better not to change the default value.
            nb_steps: Number of training steps/iterations in an epoch
            scale_min_loss: A value that is used to scale the minimum loss found yet to compare it with the
                            current loss. A value between 3-5 is found to be nominal

            clip_initial_values: How many initial values to clip in the plots?
            clip_end_values: How many values to clip from the end in the plots?
        """
        super().__init__()
        assert nb_steps > 1, "Number of training steps should be greateer than 1"

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.beta = beta
        self.nb_steps = nb_steps
        self.min_loss = 0
        self.batch_num = 0
        self.avg_loss = 0
        self._lr_scaler = (max_lr / base_lr) ** (1 / (nb_steps - 1))
        self.min_loss_scaler = scale_min_loss
        self.clip_initial_values = clip_initial_values
        self.clip_end_values = clip_end_values
        self.history = {}

    @property
    def lr_scaler(self):
        return (self.max_lr / self.base_lr) ** (1 / (self.nb_steps - 1))

    def on_train_begin(self, logs={}):
        logs = logs or {}
        if self.batch_num == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)

    def on_batch_begin(self, epoch, logs=None):
        self.batch_num += 1
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs["loss"]
        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * loss
        loss_smoothened = self.avg_loss / (1 - self.beta ** self.batch_num)

        if self.batch_num == 1 or loss_smoothened < self.min_loss:
            self.min_loss = loss_smoothened

        self.history.setdefault('batch_loss', []).append(loss_smoothened)
        self.history.setdefault('lr', []).append(math.log10(self.base_lr))

        if self.min_loss_scaler:
            max_loss_check = self.min_loss_scaler * self.min_loss
        else:
            max_loss_check = 4 * self.min_loss

        if self.batch_num > 1 and (np.isnan(loss_smoothened)
                                   or loss_smoothened > max_loss_check):
            print("LR range test complete")
            self.model.stop_training = True
            # self.plot_values(self.history['lr'], self.history['batch_loss'])

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.base_lr *= self.lr_scaler
        K.set_value(self.model.optimizer.lr, self.base_lr)

    def plot_values(self, x, y):
        f, ax = plt.subplots(1, 1, figsize=(8, 5))
        if self.clip_initial_values:
            x = x[self.clip_initial_values:]
            y = x[self.clip_initial_values:]

        if self.clip_end_values:
            x = x[:self.clip_end_values]
            y = y[:self.clip_end_values]

        ax.plot(x, y)
        plt.title("LR_range test")
        plt.xlabel("Learning rate on log10 scale")
        plt.ylabel("Loss")
        plt.show()

