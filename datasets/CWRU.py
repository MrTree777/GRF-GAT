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
root='E:\PythonProject\DataSets\CW_DataSet'
#0、1、2和3分别代表四种不同的工况     dataname是一个字典，其中的每个键对应着一个包含文件名的列表
dataname= {0:["97.mat","105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat","234.mat"],  # 1797rpm
           1:["98.mat","106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat","235.mat"],  # 1772rpm
           2:["99.mat","107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat","236.mat"],  # 1750rpm
           3:["100.mat","108.mat", "121.mat","133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat","237.mat"]}  # 1730rpm
"""上述data对应的label为
    正常数据、0.007直径内圈、0.007球、0.007以载荷区为中心、0.014直径内圈、0.014球、0.014以载荷区为中心、0.021直径内圈、0.021球、0.021以载荷区为中心
    0、1、2、3、4、5、6、7、8、9
    共10个label
"""

datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
               "Normal Baseline Data"]

"""
DE - drive end accelerometer data   驱动端加速度计数据
FE - fan end accelerometer data     风扇端加速度计数据
BA - base accelerometer data        底座加速度计数据
"""
axis = ["_DE_time", "_FE_time", "_BA_time"]

label = [i for i in range(0, 10)]   #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


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
        for n in tqdm(range(len(dataname[N[k]]))):  #这里迭代10次 取出10个.mat文件 如果n==0 取第一列normal数据，反之取12k的数据 长度是10
            #根据n的取值决定数据集的路径
            if n==0:
                #获得的是正常基线数据 字典中第一列的文件
               path1 =os.path.join(root,datasetname[3], dataname[N[k]][n])  #/Users/chensheng/Desktop/CW_DataSet/Normal Baseline Data/
            else:
                #获得的是12k驱动端轴承故障数据 字典中第一列后面的文件
                path1 = os.path.join(root,datasetname[0], dataname[N[k]][n])#/Users/chensheng/Desktop/CW_DataSet/12k Drive End Bearing Fault Data/
            data1, lab1 = data_load(path1,dataname[N[k]][n],label=label[n]) #切分数据 打上对应标签
            data += data1   #对应数据
            lab +=lab1      #对应标签
    #返回data列表 和 lab列表
    return [data, lab]

#数据加载
def data_load(filename, axisname, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")    #以.为分隔符分割字符串
    if eval(datanumber[0]) < 100:       #对数据文件名分割 取[0]位置上的值 若小于100则为normal数据
        realaxis = "X0" + datanumber[0] + axis[0]
    else:                               #因为.mat文件里是以 X+num+加速计端位命名 所以做此拼接 从而得到
        realaxis = "X" + datanumber[0] + axis[0]
    fl = loadmat(filename)[realaxis]    #通过绝对路径加载.mat文件，通过变量名获取具体内容，返回的是一个Numpy数据
    data = []
    lab = []
    start, end = 0, signal_size         #赋值 start = 0 end = signal_size = 1024
    #使用start和end将从mat文件中读取的数据切片并存储到列表data和lab中  每个切片长度为siganl_size  label为指定的值
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += 512
        end += 512
    return data, lab

#--------------------------------------------------------------------------------------------------------------------
class CWRU(object):
    num_classes = 10
    inputchannel = 1
    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]    #transfer_task默认是[[2], [1]] 故source_N == [2]
        self.target_N = transfer_task[1]    #target_N == [1]    是列表     对应的是工况1 和 2 论文做的是1和2跨域任务
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




