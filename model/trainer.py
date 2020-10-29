#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: trainer.py
    Description:
    
Created by YongBai on 2020/10/14 1:48 PM.
"""


import os
import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import recall_score, precision_score, accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:

    def __init__(self, f_model_path, csv_metrics_path,
                 nb_epochs=100, lr=0.0001, gpus=None, class_wieght=None):

        self.nb_epochs = nb_epochs
        self.f_model_path = f_model_path
        self.csv_metrics_path = csv_metrics_path
        self.lr = lr
        self.gpus = gpus
        self.class_weight = class_wieght

    def train(self, model, train_dataset, val_dataset):

        model = nn.DataParallel(model, device_ids=self.gpus) \
            if torch.cuda.is_available() and torch.cuda.device_count() > 1 and len(self.gpus) > 0 else model
        model.to(device)

        cls_weight = torch.tensor([self.class_weight[0], self.class_weight[1]], dtype=torch.float32).to(device)

        # loss_fn = nn.CrossEntropyLoss(weight=cls_weight)
        loss_fn_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight[1], dtype=torch.float32).to(device))
        # loss_fn_mse = nn.MSELoss()

        # opt = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-5)
        opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr, momentum=0.9, weight_decay=1e-5)
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, verbose=True)
        lr_scheduler = optim.lr_scheduler.CyclicLR(opt, base_lr=0.0001, max_lr=0.1,
                                                   step_size_up=150, step_size_down=150)

        # for param in model.encoder.parameters():
        #     print(param.requires_grad)

        val_loss_min = np.Inf
        csv_metrics_header = 'epoch,train_loss,val_loss,train_recall,val_recal,train_prec,val_prec,train_acc,val_acc'
        if os.path.exists(self.csv_metrics_path):
            os.remove(self.csv_metrics_path)
        with open(self.csv_metrics_path, 'a') as f:
            f.write("{}\n".format(csv_metrics_header))

        alpha = 2.0

        for epoch in range(self.nb_epochs):

            epoch_str = 'Epoch {}/{}'.format(epoch + 1, self.nb_epochs)
            # training
            model.train()
            train_loss = 0.0
            total_batchs = len(train_dataset) - 1

            for batch_idx, trn_batch_dataset in enumerate(train_dataset):
                # if batch_idx == total_batchs:
                #     continue
                # clear gradients for next train
                opt.zero_grad()

                batch_x, batch_y = trn_batch_dataset

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).unsqueeze(1)

                # enc_x, out_x = model(batch_x)
                pred_y_logit = model(batch_x)

                # calculate loss
                loss = loss_fn_bce(pred_y_logit, batch_y)
                # loss_bce = loss_fn_bce(pred_y_logit, batch_y)

                # loss = alpha * loss_mse + (1 - alpha) * loss_bce
                # loss = loss_mse + alpha * loss_bce
                # loss backward
                loss.backward()
                # update parameters
                opt.step()
                train_loss = loss.item()

                # pred_y_sigmod = torch.sigmoid(pred_y_logit).detach().cpu().numpy()
                # pred_y_ = (pred_y_sigmod >= 0.5).astype(np.int32)
                # true_y_ = batch_y.cpu().numpy()
                #
                # train_recall = recall_score(true_y_, pred_y_)
                # train_precision = precision_score(true_y_, pred_y_)
                # train_acc = accuracy_score(true_y_, pred_y_)
            # train_loss = train_loss / len(train_dataset)
                logging.info('{},[{}/{}]:tr_loss={:.4f}'.format(
                    epoch_str, batch_idx+1, len(train_dataset), train_loss))

            # validation
            model.eval()
            val_loss = 0.0
            val_recall = 0.0
            val_precision = 0.0
            val_acc = 0.0
            bce_loss = 0.0
            with torch.no_grad():
                for batch_idx, (val_batch_x, val_batch_y) in enumerate(val_dataset):

                    val_batch_x = val_batch_x.to(device)
                    val_batch_y = val_batch_y.to(device).unsqueeze(1)
                    val_pred_y = model(val_batch_x)

                    # print(val_batch_y)
                    # print(val_pred_y)

                    # calculate loss
                    # val_loss_mse = loss_fn_mse(val_out_x, val_batch_x)
                    val_loss_bce = loss_fn_bce(val_pred_y, val_batch_y)

                    # val_loss_item = alpha * val_loss_mse + (1 - alpha) * val_loss_bce
                    # val_loss_item = val_loss_mse + alpha * val_loss_bce

                    val_loss += val_loss_bce.item()

                    # bce_loss += val_loss_bce.item()
                    # val_pred_y_ = torch.sigmoid(val_pred_y).cpu().numpy()
                    # val_pred_y_ = (val_pred_y_ >= 0.5).astype(np.int32)
                    # val_true_y_ = val_batch_y.cpu().numpy()
                    #
                    # val_recall += recall_score(val_true_y_, val_pred_y_)
                    # val_precision += precision_score(val_true_y_, val_pred_y_)
                    # val_acc += accuracy_score(val_true_y_, val_pred_y_)

            val_loss = val_loss / len(val_dataset)
            bce_loss = bce_loss / len(val_dataset)
            # val_recall = val_recall / len(val_dataset)
            # val_precision = val_precision / len(val_dataset)
            val_acc = val_acc / len(val_dataset)
            logging.info('{}: val_loss={:.4f},val_recall={:.4f},val_prec={:.4f},val_acc={:.4f},val_bce_loss={:.4f}'.format(
                epoch_str, val_loss, val_recall, val_precision, val_acc, bce_loss))

            if val_loss < val_loss_min:
                last_val_loss = val_loss_min
                val_loss_min = val_loss
                # val_loss_min = val_loss
                # save on GPU
                if torch.cuda.is_available() and torch.cuda.device_count() > 1 and len(self.gpus) > 0:
                    checkpoint = {
                        'epoch': epoch + 1,
                        'valid_loss_min': val_loss,
                        'state_dict': model.module.state_dict(),
                        'optimizer': opt.state_dict(),
                    }
                else:
                    checkpoint = {
                        'epoch': epoch + 1,
                        'valid_loss_min': val_loss,
                        'state_dict': model.state_dict(),
                        'optimizer': opt.state_dict(),
                    }

                if os.path.exists(self.f_model_path):
                    os.remove(self.f_model_path)
                torch.save(checkpoint, self.f_model_path)
                logging.info('model performance improved from {} to {}, best model saved at {}'.format(
                    last_val_loss, val_loss_min, self.f_model_path))
            else:
                logging.info('model performance not improved')

            # save train metrics
            with open(self.csv_metrics_path, 'a') as f:
                f.write("{},{},{},{},{},{}\n".format(epoch + 1, train_loss, val_loss,
                                                              val_recall, val_precision, val_acc))

            # lr_scheduler.step(val_loss)
            lr_scheduler.step()

    def predict(self, model, tst_dataset, out_value_path=None):
        checkpoint = torch.load(self.f_model_path)
        # model = nn.DataParallel(model, device_ids=self.gpus) \
        #     if torch.cuda.is_available() and torch.cuda.device_count() > 1 else model
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)

        if out_value_path is not None and os.path.exists(out_value_path):
            os.remove(out_value_path)

        model.eval()
        pred_y = torch.tensor([], device=device)
        true_y = torch.tensor([], device=device)

        with torch.no_grad():
            for batch_idx, (tst_batch_phage_x, tst_batch_bact_x, tst_batch_dataset_y) in enumerate(tst_dataset):
                tst_batch_phage_x = tst_batch_phage_x.to(device)
                tst_batch_bact_x = tst_batch_bact_x.to(device)
                tst_batch_dataset_y = tst_batch_dataset_y.to(device)

                tst_pred_y = model(tst_batch_phage_x, tst_batch_bact_x).squeeze(-1)

                pred_y = torch.cat((pred_y, tst_pred_y), 0)
                true_y = torch.cat((true_y, tst_batch_dataset_y), 0)

                logging.info('Batch {}/{}, prediction DONE'.format(batch_idx+1, len(tst_dataset)))
        out_df = pd.DataFrame(data={'pred_y': pred_y.cpu().numpy(), 'true_y': true_y.cpu().numpy()})
        out_df.to_csv(out_value_path, index=False)
        logging.info('prediction DONE.')