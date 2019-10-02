# -*- coding:utf-8 -*-
from numpy import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LanNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, bn_dim=100, output_dim=10):
        super(LanNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bn_dim = bn_dim
        self.output_dim = output_dim

        self.layer0 = nn.Sequential( 
             nn.Conv2d(in_channels=1, out_channels=16, stride=1, kernel_size=5,padding=2),
             nn.BatchNorm2d(16),
             nn.ReLU(),
             nn.MaxPool2d(2)
             )
        self.layer01 = nn.Sequential(
             nn.Conv2d(in_channels=16, out_channels=32, stride=1, kernel_size=5,padding=2),
             nn.BatchNorm2d(32),
             nn.ReLU(),
             nn.MaxPool2d(2)
                  )
#             nn.Conv2d(in_channels=16, out_channels=1, stride=1, kernel_size=5,padding=2))
        self.layer00 = nn.Sequential(torch.nn.Linear(self.input_dim,512),
              torch.nn.ReLU(),
              torch.nn.Linear(512, 40))

        self.layer1 = nn.Sequential()
        self.layer1.add_module('lstm', nn.LSTM(self.input_dim, self.hidden_dim/2, num_layers=2, batch_first=True, bidirectional=True))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('linear', nn.Linear(163840, self.bn_dim))
        # self.layer2.add_module('Sigmoid', nn.Sigmoid())

        self.layer3 = nn.Sequential()
        self.layer3.add_module('linear', nn.Linear(self.bn_dim, self.output_dim))

    def forward(self, src, mask, target):
        batch_size, fea_frames, fea_dim = src.size()
        print(shape(src))
        srcor = src
        src = src.reshape(src.shape[0],1, src.shape[1], src.shape[2])
        print(shape(src))
        out_hidden = self.layer0(src)
        out_hidden = self.layer01(out_hidden)
#        out_hidden.view(out_hidden.size(0), -1)
		#########dnn feature###########
        #nnf = self.layer00(srcor)
		####################
        print(shape(out_hidden))
        out_hidden = out_hidden.reshape(out_hidden.shape[0],out_hidden.shape[1]*out_hidden.shape[2], out_hidden.shape[3])
#        out_hidden0 = np.concatenate((out_hidden0, nnf), axis=2)
        #out_hidden0 = out_hidden0 + nnf
        #out_hidden, hidd = self.layer1(out_hidden0)
        print(shape(out_hidden))
        out_hidden = out_hidden.contiguous().view(-1, out_hidden.size(-1))   
        print(shape(out_hidden))
        out_bn = self.layer2(out_hidden)
        out_target = self.layer3(out_bn)


        out_target = out_target.contiguous().view(batch_size, fea_frames, -1)
        mask = mask.contiguous().view(batch_size, fea_frames, 1).expand(batch_size, fea_frames, out_target.size(2))
        out_target_mask = out_target * mask
        out_target_mask = out_target_mask.sum(dim=1)/mask.sum(dim=1)
        predict_target = F.softmax(out_target_mask, dim=1)

        # 计算loss
        tar_select_new = torch.gather(predict_target, 1, target)
        ce_loss = -torch.log(tar_select_new) 
        ce_loss = ce_loss.sum() / batch_size

        # 计算acc
        (data, predict) = predict_target.max(dim=1)
        predict = predict.contiguous().view(-1,1)
        correct = predict.eq(target).float()       
        num_samples = predict.size(0)
        sum_acc = correct.sum().item()
        acc = sum_acc/num_samples

        return acc, ce_loss
