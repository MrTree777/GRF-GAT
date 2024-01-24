import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm
# --------------------------------获取数据-----------------------------
#Digital data was collected at 12,000 samples per second
#单个样本长度为1024
signal_size = 1024
root='E:\PythonProject\DataSets\SQ_DataSet'
#0、1、2分别代表四种不同的工况     dataname是一个字典，其中的每个键对应着一个包含文件名的列表
dataname= {0:["NC_30hz.mat","IR1_30hz.mat", "IR2_30hz.mat", "IR3_30hz.mat","OR1_30hz.mat","OR2_30hz.mat","OR3_30hz.mat"],  # 30rpm
           1:["NC_40hz.mat","IR1_40hz.mat", "IR2_40hz.mat", "IR3_40hz.mat","OR1_40hz.mat","OR2_40hz.mat","OR3_40hz.mat"],  # 40rpm
           2:["NC_20hz.mat","IR1_20hz.mat", "IR2_20hz.mat", "IR3_20hz.mat","OR1_20hz.mat","OR2_20hz.mat","OR3_20hz.mat"],  # 20rpm
           3:["NC_10hz.mat","IR1_10hz.mat", "IR2_10hz.mat", "IR3_10hz.mat","OR1_10hz.mat","OR2_10hz.mat","OR3_10hz.mat"],  # 10rpm
}
"""上述data对应的label为
    正常数据、内圈、外圈
    0、1、2、3、4、5、6
    共7个label
"""


label = [i for i in range(0, 7)]   #[0, 1, 2, 3]


def get_files(root, N):#N为转速，N传进来是一个代表负载
    '''
    This function is used to generate the final training set and test set.  用来生成训练和测试数据集
    root:The location of the data set                                       参数root是数据集的路径
    '''
    data = []
    lab =[]
    #k 是从 0 到 len(N) - 1 的整数，用于遍历 N 列表中的元素。N 列表中的每个元素代表着一种负载状态，也就是转速
    for k in range(len(N)): #k只取一个值 0
        #N[k] 是列表 N 中的第 k 个元素，表示转速值     dataname[N[k]] 代表着对应转速值下的所有文件名列表
        for n in tqdm(range(len(dataname[N[k]]))):  #
            #根据n的取值决定数据集的路径
            if n==0:
                #获得的是正常基线数据 字典中第一列的文件
               path1 =os.path.join(root,dataname[N[k]][n])  #
            else:
                #获得的是故障数据 字典中第一列后面的文件
                path1 = os.path.join(root, dataname[N[k]][n])#
            data1, lab1 = data_load(path1,label=label[n]) #切分数据 打上对应标签
            data += data1   #对应数据
            lab +=lab1      #对应标签
    #返回data列表 和 lab列表
    return [data, lab]

#数据加载
def data_load(filename, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = loadmat(filename)['data']    #通过绝对路径加载.mat文件，通过变量名获取具体内容，返回的是一个Numpy数据
    data = []
    lab = []
    start, end = 0, signal_size         #赋值 start = 0 end = signal_size = 1024
    #使用start和end将从mat文件中读取的数据切片并存储到列表data和lab中  每个切片长度为siganl_size  label为指定的值
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += 1024
        end += 1024

    return data, lab

#--------------------------------------------------------------------------------------------------------------------
class SQ(object):
    num_classes = 7
    inputchannel = 1
    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]    #transfer_task设置为[[2], [1]] 故source_N == [2]
        self.target_N = transfer_task[1]    #target_N == [1]    是列表     对应的是工况1 和 2
        self.normlizetype = normlizetype
        """
            下述代码定义了一个数据增强的管道，包括两个阶段，训练和验证阶段
            在训练阶段，数据增强管道包括Respe、归一化、Retype等操作，可以对输入数据进行形状重塑、归一化等预处理操作
            同时还可以对输入的数据进行随机添加高斯噪声、随机缩放、随机拉伸、随机裁剪等操作，以扩充数据集，增加模型的泛化性
            在验证阶段，数据增强管道只包括 Reshape、Normalize、Retype 等预处理操作，不进行数据增强操作，以保证验证集的纯净性
            通过定义 self.data_transforms，将不同的数据预处理操作分别应用于训练集和验证集
            在训练过程中，每个batch的数据会被按照self.data_transforms['train']的定义进行预处理操作
            在验证过程中，则按照self.data_transforms['val']的定义进行预处理操作
        """
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                # Scale(1)
            ])
        }

    """
        这个函数是将数据集分割成源域训练集、源域验证集、目标域训练集和目标域验证集，用于域自适应算法
        在域自适应中，需要使用源域数据进行模型的训练和验证，再使用目标域数据进行模型的测试，因此需要将数据集分割成这些部分
        其中，训练集用于模型的训练，验证集用于模型的验证，目标域的验证集则是用来测试模型的泛化能力，检查模型是否过拟合了
    """
    def data_split(self, transfer_learning=True):
        #此处是迁移学习
        if transfer_learning:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            #将list_data的数据存储到一个表格中
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            #将数据集分割成训练和验证集 不同标签在训练和验证集所占比例相同 80%的训练集和20%的验证集
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            #通过PyTorch的数据集类加载源域训练数据 得到序列数据和对应标签
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            # 通过PyTorch的数据集类加载源域验证数据 得到序列数据和对应标签
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            # 划分训练集和剩余数据集
            train_pd, remaining_pd = train_test_split(data_pd, test_size=0.4, random_state=40,
                                                      stratify=data_pd["label"])
            # 划分验证集和测试集
            val_pd, test_pd = train_test_split(remaining_pd, test_size=0.5, random_state=40,
                                               stratify=remaining_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            target_test = dataset(list_data=test_pd, transform=self.data_transforms['val'])

            return source_train, source_val, target_train, target_val,target_test
        #此处是领域自适应
        else:
            #get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_val



