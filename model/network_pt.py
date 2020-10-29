#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: network_pt.py
    Description:
    
Created by YongBai on 2020/10/14 1:05 AM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhageBactNet(nn.Module):

    def __init__(self, nb_feats, dropout=0.0, embed_bool=True):
        super(PhageBactNet, self).__init__()

        phage_convout_channels = 32
        bact_convout_channels = 128
        self.embed_bool = embed_bool
        if self.embed_bool:
            self.emb = nn.Embedding(6, nb_feats)

        # phage
        self.phage_conv1 = self.__block(nb_feats, 8, 16, 2, dropout)
        self.phage_conv2 = self.__block(8, 8, 16, 2, dropout)

        self.phage_conv3 = self.__block(8, 64, 16, 1, dropout)
        self.phage_conv4 = self.__block(64, 64, 16, 1, dropout)

        self.phage_conv5 = self.__block(64, 32, 1, 1, dropout)
        self.phage_conv6 = self.__block(32, phage_convout_channels, 1, 1, dropout)

        # bact
        self.bact_conv1 = self.__block(nb_feats, 16, 16, 8, dropout)
        self.bact_conv2 = self.__block(16, 16, 16, 8, dropout)

        self.bact_conv3 = self.__block(16, 64, 16, 1, dropout)
        self.bact_conv4 = self.__block(64, 64, 16, 1, dropout)

        self.bact_conv5 = self.__block(64, 128, 1, 1, dropout)
        self.bact_conv6 = self.__block(128, bact_convout_channels, 1, 1, dropout)

        # self.conv1_trans = nn.ConvTranspose1d(in_channels=phage_convout_channels * 2 + bact_convout_channels * 2,
        #                                       out_channels=8,
        #                                       kernel_size=8)
        # # nn.init.kaiming_normal_(self.conv1_trans.weight)
        #
        # self.lstm_hidden_size = 16
        # self.lstm_nb_hidden_layers = 2
        #
        # self.stackedlstm = nn.LSTM(input_size=8,
        #                            hidden_size=self.lstm_hidden_size,
        #                            dropout=dropout,
        #                            num_layers=self.lstm_nb_hidden_layers,
        #                            bidirectional=True, batch_first=True)
        #
        # # here multiple 2 is becuase bidirectional = True
        # self.fc_1 = nn.Linear(self.lstm_hidden_size * 2, 128)
        # self.fc_1 = nn.Linear(phage_convout_channels * 2 + bact_convout_channels * 2, 128)
        # self.fc_2 = nn.Linear(128, 1)

        self.fc_1 = nn.Linear(phage_convout_channels * 2 + bact_convout_channels * 2, 1)

    def forward(self, phage_input, bact_input):

        if self.embed_bool:
            phage_emb = self.emb(phage_input)
            phage_emb = phage_emb.permute(0, 2, 1)
            bact_emb = self.emb(bact_input)
            bact_emb = bact_emb.permute(0, 2, 1)
        else:
            phage_emb = phage_input
            bact_emb = bact_input
        # phage
        phage_x = self.phage_conv1(phage_emb)
        phage_x = self.phage_conv2(phage_x)
        phage_x = F.max_pool1d(phage_x, kernel_size=2)

        phage_x = self.phage_conv3(phage_x)
        phage_x = self.phage_conv4(phage_x)
        phage_x = F.max_pool1d(phage_x, kernel_size=2)

        phage_x = self.phage_conv5(phage_x)
        phage_x = self.phage_conv6(phage_x)
        phage_x_max = F.max_pool1d(phage_x, kernel_size=phage_x.shape[-1])
        phage_x_avg = F.avg_pool1d(phage_x, kernel_size=phage_x.shape[-1])
        phage_x = torch.cat((phage_x_max, phage_x_avg), dim=1)

        #bact
        bact_x = self.bact_conv1(bact_emb)
        bact_x = self.bact_conv2(bact_x)
        bact_x = F.max_pool1d(bact_x, kernel_size=4)

        bact_x = self.bact_conv3(bact_x)
        bact_x = self.bact_conv4(bact_x)
        bact_x = F.max_pool1d(bact_x, kernel_size=4)

        bact_x = self.bact_conv5(bact_x)
        bact_x = self.bact_conv6(bact_x)
        bact_x_max = F.max_pool1d(bact_x, kernel_size=bact_x.shape[-1])
        bact_x_avg = F.avg_pool1d(bact_x, kernel_size=bact_x.shape[-1])
        bact_x = torch.cat((bact_x_max, bact_x_avg), dim=1)

        x = torch.cat((phage_x, bact_x), dim=1).squeeze()

        # x = self.conv1_trans(x)  # output x.size = [batch, features, seq_len]
        # x = x.permute(0, 2, 1)   # output x.size = [batch, seq_len, features]
        #
        # # h_0 = torch.zeros(self.lstm_nb_hidden_layers * 2, x.size(0), self.lstm_hidden_size).requires_grad_()
        # # c_0 = torch.zeros(self.lstm_nb_hidden_layers * 2, x.size(0), self.lstm_hidden_size).requires_grad_()
        # #
        # # h_0 = h_0.cuda()
        # # c_0 = c_0.cuda()
        # # x, _ = self.stackedlstm(x, (h_0.detach(), c_0.detach()))
        #
        # x, _ = self.stackedlstm(x)
        # # output x.shape = [batch, seq_len, features]
        # # select the latest timestep output from lstm
        # x = F.leaky_relu(self.fc_1(x[:, -1, :]))
        # x = F.leaky_relu(self.fc_1(x))
        # out = torch.softmax(self.fc_2(x), dim=-1)
        out = torch.sigmoid(self.fc_1(x))
        return out

    def __block(self, nb_in_channel, nb_out_channel, kernel_size, stride, dropout):
        conv = nn.Sequential(
            nn.Conv1d(
                in_channels=nb_in_channel,
                out_channels=nb_out_channel,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(nb_out_channel),
            nn.ReLU(True)
            # nn.LeakyReLU()
        )
        # nn.init.kaiming_normal_(conv[0].weight)
        # nn.init.xavier_uniform_(conv[0].weight, gain=nn.init.calculate_gain('relu'))
        return conv
