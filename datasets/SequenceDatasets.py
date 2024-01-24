#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from datasets.sequence_aug import *


"""
    下述代码是一个自定义的PyTorch 数据集类 dataset，用于加载训练数据和测试数据
    在 __init__ 函数中，通过 list_data 参数传入数据
    如果 test=True，表示数据是测试数据，只包含序列数据，没有标签
    如果 test=False，表示数据是训练数据，包含序列数据和对应的标签
    这里通过 tolist() 方法将 Pandas DataFrame 数据转换成 Python 列表
    
    
    在 __getitem__ 函数中，根据索引 item，取出序列数据和对应的标签（如果是训练数据）
    通过数据管道 transforms 对序列数据进行预处理
    如果是测试数据，返回序列数据和该数据在原始数据集中的索引 item；如果是训练数据，返回序列数据和对应的标签
    
    在 __len__ 函数中，返回数据集的长度，即数据集中的数据条目数量
"""
class dataset(Dataset):

    def __init__(self, list_data, test=False, transform=None):
        self.test = test
        #训练数据带标签 测试数据不带标签
        if self.test:
            self.seq_data = list_data['data'].tolist()
        else:
            self.seq_data = list_data['data'].tolist()
            self.labels = list_data['label'].tolist()
        #通过数据管道进行数据增强
        if transform is None:
            self.transforms = Compose([
                Reshape()
            ])
        else:
            self.transforms = transform


    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        if self.test:
            seq = self.seq_data[item]
            seq = self.transforms(seq)
            return seq, item
        else:
            seq = self.seq_data[item]
            label = self.labels[item]
            seq = self.transforms(seq)
            return seq, label

