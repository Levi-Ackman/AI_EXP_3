from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        if self.args.optimizer=='Adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer=='SGD':
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer=='RMSprop':
            model_optim = optim.RMSprop(self.model.parameters(), lr=self.args.learning_rate)
        
        return model_optim

    def _select_criterion(self):
        if self.args.loss=='mse':
            criterion = nn.MSELoss()
        elif self.args.loss=='mae':
            criterion = nn.L1Loss()
        return criterion

    def _vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, -1]
                batch_y = batch_y[:, -self.args.pred_len:, -1].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        # x, y, x_mark, y_mark=next(iter(train_loader))
        # import the dataset

        path = os.path.join(self.args.checkpoints, 'seq_'+str(self.args.seq_len)+'_label_'+str(self.args.label_len)+'_pre_'+str(self.args.pred_len)+'/'+self.args.model \
            + '/'+'opt_'+self.args.optimizer +'/'+'bz_'+str(self.args.batch_size)+'/'+'lr_'+str(self.args.learning_rate)+'/')
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                # print('tensor shape:',batch_x.shape, batch_x_mark.shape, dec_inp.shape, batch_y_mark.shape)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs[:, -self.args.pred_len:, -1]
                        batch_y = batch_y[:, -self.args.pred_len:, -1].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs[:, -self.args.pred_len:, -1]
                    batch_y = batch_y[:, -self.args.pred_len:, -1].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # if (i + 1) % 100 == 0:
                #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self._vali(vali_data, vali_loader, criterion)
            test_loss = self._vali(test_data, test_loader, criterion)

            print("Epoch: {}, Steps: {} | Train Loss: {} Vali Loss: {} Test Loss: {}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        path=os.path.join(self.args.checkpoints, 'seq_'+str(self.args.seq_len)+'_label_'+str(self.args.label_len)+'_pre_'+str(self.args.pred_len)+'/'+self.args.model)
        best_model_path = path + '/' +'opt_'+self.args.optimizer+'/'+'bz_'+str(self.args.batch_size)+'_lr_'+str(self.args.learning_rate)+ 'checkpoint.pth'

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        preds = []
        trues = []
        folder_path = './test_results/'+'seq_'+str(self.args.seq_len)+'_label_'+str(self.args.label_len)+'_pre_'+str(self.args.pred_len)+'/'+self.args.model+'/'+'opt_'+self.args.optimizer +'/'+ 'bz_'+str(self.args.batch_size)+'/'+'lr_'+str(self.args.learning_rate)+ '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, -1]
                batch_y = batch_y[:, -self.args.pred_len:, -1].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % self.args.seg == 0:
                    gt=true[0]
                    pd=pred[0]
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        preds = np.array(preds)
        trues = np.array(trues)
        
        preds=np.concatenate(preds, axis=0) 
        trues=np.concatenate(trues, axis=0) 
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
        my_dict = {
            'model':self.args.model,
            'batch_size':str(self.args.batch_size),
            'lr':str(self.args.learning_rate),
            'Optim':self.args.optimizer,
            'mse':float(mse), 
            'mae':float(mae), 
            'rmse':float(rmse),
            'mape':float(mape),
            'mspe':float(mspe),
               }
        dict_path='./test_dict/'+'seq_'+str(self.args.seq_len)+'_label_' \
                  +str(self.args.label_len)+'_pre_'+str(self.args.pred_len)+'/'+self.args.model+'/'+'opt_'+self.args.optimizer+'/'  \
                  +'bz_'+str(self.args.batch_size)+'/'+'lr_'+str(self.args.learning_rate)
                  
        if not os.path.exists(dict_path):
            os.makedirs(dict_path)
        with open(os.path.join(dict_path, 'records.json'), 'w') as f:
             json.dump(my_dict, f)
        f.close()
        return
