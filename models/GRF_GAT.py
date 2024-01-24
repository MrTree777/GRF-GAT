#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings
import torch



from models.GATConv import MH_GAT
from models.WDCNN import WDCNN



class GRF_GAT_features(nn.Module):
    def __init__(self, pretrained=False):
        super(GRF_GAT_features, self).__init__()
        self.model_wdcnn = WDCNN()
        #这里加注意力机制
        self.model_GATConv = MH_GAT()
        self.__in_features = 256*1

    def forward(self, x):
        x1 = self.model_wdcnn(x)
        x2 = self.model_GATConv(x1)


        return x2

    def output_num(self):
        return self.__in_features