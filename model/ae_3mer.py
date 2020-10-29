#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: ae_3mer.py
    Description:
    
Created by YongBai on 2020/10/22 7:37 PM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


NB_FEATS = 5 ** 3
SEQ_LEN = 11000
DROPOUT = 0.3


class AEEncoder3Mer(nn.Module):
    def __init__(self):

        super(AEEncoder3Mer, self).__init__()

        self.conv0 = nn.Conv1d(in_channels=NB_FEATS, out_channels=128, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm1d(128)

        self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)

        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(32)

        self.conv5 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(16)

    def forward(self, x):

        x = F.relu(self.bn0(self.conv0(x)))
        x = F.dropout(x, DROPOUT, self.training)
        x = F.max_pool1d(x, 2)                  # [32, 128, 5500]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.dropout(x, DROPOUT, self.training)
        x = F.max_pool1d(x, 2)                  # [32, 64, 2750]

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.dropout(x, DROPOUT, self.training)
        x = F.max_pool1d(x, 2)                  # [32, 32, 1375]

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.dropout(x, DROPOUT, self.training)
        x = F.max_pool1d(x, 2)                  # [32, 32, 687]

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.dropout(x, DROPOUT, self.training)
        x = F.max_pool1d(x, 2)                  # [32, 32, 343]

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.dropout(x, DROPOUT, self.training)
        x = F.max_pool1d(x, 2)                  # [32, 16, 171]

        return x


class AEDecoder3Mer(nn.Module):

    def __init__(self):
        super(AEDecoder3Mer, self).__init__()
        self.conv0 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm1d(32)

        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv5 = nn.Conv1d(in_channels=128, out_channels=NB_FEATS, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(NB_FEATS)

        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        """

        Parameters
        ----------
        x (N, 16, 171)

        Returns
        -------

        """

        x = F.relu(self.bn0(self.conv0(x)))                                 # out (N, 32, 171)
        x = F.dropout(x, DROPOUT, self.training)
        x = self.upsample(x)                                                # out (N, 32, 342)

        x = F.pad(x, (0, 1), mode='replicate')
        x = F.relu(self.bn1(self.conv1(x)))                                 # out (N, 32, 343)
        x = F.dropout(x, DROPOUT, self.training)
        x = self.upsample(x)                                                # out (N, 32, 686)

        x = F.pad(x, (0, 1), mode='replicate')                              # out (N, 32, 687)
        x = F.relu(self.bn2(self.conv2(x)))                                 # out (N, 32, 687)
        x = F.dropout(x, DROPOUT, self.training)
        x = self.upsample(x)                                                # out (N, 32, 1374)

        x = F.pad(x, (0, 1), mode='replicate')                              # out (N, 32, 1375)
        x = F.relu(self.bn3(self.conv3(x)))                                 # out (N, 64, 1375)
        x = F.dropout(x, DROPOUT, self.training)
        x = self.upsample(x)                                                # out (N, 64, 2750)

        x = F.relu(self.bn4(self.conv4(x)))                                 # out (N, 128, 2750)
        x = F.dropout(x, DROPOUT, self.training)
        x = self.upsample(x)                                                # out (N, 128, 5500)

        x = F.relu(self.bn5(self.conv5(x)))                                 # out (N, 125, 5500)
        x = F.dropout(x, DROPOUT, self.training)
        x = self.upsample(x)                                                # out (N, 125, 11000)

        return x


class PhageBactAutoEncoder(nn.Module):

    def __init__(self):
        super(PhageBactAutoEncoder, self).__init__()
        self.encoder = AEEncoder3Mer()
        self.decoder = AEDecoder3Mer()

    def forward(self, x):

        enc_x = self.encoder(x)  # out (N, 8, 371)
        out_x = self.decoder(enc_x)

        return enc_x, out_x


class AECls(nn.Module):

    def __init__(self):
        super(AECls, self).__init__()

        self.encoder = AEEncoder3Mer()   # out: (N, 16, 171)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 171, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(500, 1),
        )

    def forward(self, x):
        enc_x = self.encoder(x)  # out (N, 8, 371)
        cls_logit = self.classifier(enc_x)
        return cls_logit


class AEConvClsa(nn.Module):
    def __init__(self):
        super(AEConvClsa, self).__init__()

        self.encoder = AEEncoder3Mer()  # out: (N, 16, 171)

        self.classifier = nn.Sequential(

            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.MaxPool1d(2),                # out: (N, , 84)

            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.MaxPool1d(2),                # out: (N, 8, 41)

            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=1, stride=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Flatten(),
            nn.Linear(4 * 41, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),

            nn.Linear(500, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),

            nn.Linear(100, 1),
        )

    def forward(self, x):
        enc_x = self.encoder(x)
        cls_logit = self.classifier(enc_x)
        return cls_logit


class AESECls(nn.Module):
    def __init__(self):
        super(AESECls, self).__init__()

        self.encoder = AEEncoder3Mer()  # out: (N, 16, 171)

        self.conv1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1) # (16, 171)
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)  # (16, 171)
        self.bn2 = nn.BatchNorm1d(16)

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)  # (16, 85)
        self.bn3 = nn.BatchNorm1d(16)

        self.conv4 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)   # (16, 85)
        self.bn4 = nn.BatchNorm1d(16)

        self.conv5 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)  # (16, 42)
        self.bn5 = nn.BatchNorm1d(16)

        self.conv6 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)  # (16, 42)
        self.bn6 = nn.BatchNorm1d(16)

        self.se1 = SE_Block(16, r=8)
        self.se2 = SE_Block(16, r=8)
        self.se3 = SE_Block(16, r=8)

        self.maxpool = nn.MaxPool1d(2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 21, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        enc_x = self.encoder(x)
        identity = enc_x

        out = F.relu(self.bn1(self.conv1(enc_x)))
        out = self.se1(self.bn2(self.conv2(out)))
        out += identity
        out = F.relu(out)
        out = self.maxpool(out)

        identity2 = out

        out = F.relu(self.bn3(self.conv3(out)))
        out = self.se2(self.bn4(self.conv5(out)))
        out += identity2
        out = F.relu(out)
        out = self.maxpool(out)

        identity3 = out

        out = F.relu(self.bn5(self.conv5(out)))
        out = self.se3(self.bn6(self.conv6(out)))
        out += identity3
        out = F.relu(out)
        out = self.maxpool(out)

        out = self.classifier(out)

        return out


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

    def __init__(self, c, r=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, seq_len, = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1)
        return x * y.expand_as(x)
























