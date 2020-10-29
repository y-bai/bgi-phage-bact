#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: network.py
    Description:

Created by YongBai on 2020/3/18 2:09 PM.
"""
import tensorflow as tf


def _bn_relu(x, name=None):
    """
    Helper to build a BN -> relu block
    :param x:
    :param name:
    :return:
    """
    norm = tf.keras.layers.BatchNormalization(name=name + '_bn')(x)
    # return tf.keras.layers.Activation("relu", name=name + '_relu')(norm)

    # update on 20201009
    return tf.keras.activations.selu(norm)


def basic_unit(**kwargs):
    """
    Helper to build a conv1d->BN->relu residual unit
    :param res_params:
    :return:
    """

    filters = kwargs['filters']
    b_name = kwargs['name']
    kernel_size = kwargs.setdefault('kernel_size', 16)
    strides = kwargs.setdefault('strides', 1)
    l2r = kwargs.setdefault('l2r', 1e-4)
    drop_out = kwargs.setdefault('drop', 0.5)

    def f(x):
        x = tf.keras.layers.Conv1D(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   kernel_regularizer=tf.keras.regularizers.L2(l2r) if l2r is not None else None,
                                   # kernel_initializer='he_normal',
                                   name=b_name + '_conv')(x)

        x = tf.keras.layers.Dropout(drop_out)(x)  #
        return _bn_relu(x, name=b_name)
    return f


def start_block(input_phage, input_bact, l2_weight):
    phage_x = basic_unit(filters=8, kernel_size=128, strides=2, l2r=l2_weight,
                         name='phage_unit_0')(input_phage)
    phage_x = basic_unit(filters=8, kernel_size=128, strides=2, l2r=l2_weight,
                         name='phage_unit_1')(phage_x)

    phage_x = tf.keras.layers.MaxPooling1D()(phage_x)

    phage_x = basic_unit(filters=64, kernel_size=16, strides=1, l2r=l2_weight,
                         name='phage_unit_2')(phage_x)
    phage_x = basic_unit(filters=64, kernel_size=16, strides=1, l2r=l2_weight,
                         name='phage_unit_3')(phage_x)

    phage_x = tf.keras.layers.MaxPooling1D()(phage_x)

    phage_x = basic_unit(filters=32, kernel_size=1, strides=1, l2r=l2_weight,
                         name='phage_fcn_0')(phage_x)
    phage_x = basic_unit(filters=32, kernel_size=1, strides=1, l2r=l2_weight,
                         name='phage_fcn_1')(phage_x)

    phage_x_gloab_max = tf.keras.layers.GlobalMaxPooling1D()(phage_x)
    phage_x_gloab_avg = tf.keras.layers.GlobalAveragePooling1D()(phage_x)
    phage_x = tf.keras.layers.Concatenate()([phage_x_gloab_max, phage_x_gloab_avg])

    # bact
    bact_x = basic_unit(filters=16, kernel_size=512, strides=8, l2r=l2_weight,
                        name='bact_unit_0')(input_bact)
    bact_x = basic_unit(filters=16, kernel_size=512, strides=8, l2r=l2_weight,
                        name='bact_unit_1')(bact_x)

    bact_x = tf.keras.layers.MaxPooling1D()(bact_x)
    bact_x = basic_unit(filters=64, kernel_size=64, strides=1, l2r=l2_weight,
                        name='bact_unit_2')(bact_x)
    bact_x = basic_unit(filters=64, kernel_size=64, strides=1, l2r=l2_weight,
                        name='bact_unit_3')(bact_x)
    bact_x = tf.keras.layers.MaxPooling1D()(bact_x)

    bact_x = basic_unit(filters=128, kernel_size=1, strides=1, l2r=l2_weight,
                        name='bact_fcn_0')(bact_x)
    bact_x = basic_unit(filters=128, kernel_size=1, strides=1, l2r=l2_weight,
                        name='bact_fcn_1')(bact_x)
    bact_x_gloab_max = tf.keras.layers.GlobalMaxPooling1D()(bact_x)
    bact_x_gloab_avg = tf.keras.layers.GlobalAveragePooling1D()(bact_x)
    bact_x = tf.keras.layers.Concatenate()([bact_x_gloab_max, bact_x_gloab_avg])

    x = tf.keras.layers.Concatenate()([phage_x, bact_x])

    return x


def phage_bact_net_bilstm(nb_feat, l2_weight):

    input_phage = tf.keras.Input(shape=(None, nb_feat), name='input_phage')  # 349055
    input_bact = tf.keras.Input(shape=(None, nb_feat), name='input_bact')  # 5960495

    x = start_block(input_phage, input_bact, l2_weight)

    x = tf.expand_dims(x, axis=-1)
    x = tf.keras.layers.Conv1DTranspose(filters=4,
                                        kernel_size=8,
                                        strides=1,
                                        # kernel_initializer='he_normal',
                                        kernel_regularizer=tf.keras.regularizers.L2(l2_weight))(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16))(x)

    # update on 20201009
    # x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8))(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    # update on 20201009 end
    # for regression
    # output = tf.keras.layers.Dense(1, name='end_dense')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='end_dense')(x)
    model = tf.keras.Model(inputs=[input_phage, input_bact], outputs=output, name='phage_bact_net_bilstm')
    return model


def phage_bact_net_baseline(nb_feat, l2_weight):

    input_phage = tf.keras.Input(shape=(None, nb_feat), name='input_phage')  # 349055
    input_bact = tf.keras.Input(shape=(None, nb_feat), name='input_bact')  # 5960495

    x = start_block(input_phage, input_bact, l2_weight)

    x = tf.keras.layers.Dense(64, activation='relu', name='end_dense_relu_0')(x)
    # x = tf.keras.layers.Dense(16, activation='relu', name='end_dense_relu_1')(x)
    output = tf.keras.layers.Dense(1, name='end_dense_relu_2')(x)

    model = tf.keras.Model(inputs=[input_phage, input_bact], outputs=output, name='phage_bact_net')

    return model



