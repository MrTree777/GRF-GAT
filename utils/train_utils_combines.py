#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
import math
import torch
from torch import nn
from torch import optim
from utils.lr_scheduler import *
import models
import datasets
from utils.save import Save_Tool
from loss.DAN import DAN
import matplotlib.pyplot as plt

from torchviz import make_dot


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE








class train_utils(object):
    def __init__(self, args, save_dir): #两个属性
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer      初始化数据集、模型、损失和优化器
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition     考虑使用GPU还是CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count" #抛出GPU分配异常
        else:
            warnings.warn("gpu is not available")       #没有GPU，转而使用CPU
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))


        # Load the datasets     加载数据集
        Dataset = getattr(datasets, args.data_name)     #得到的是<class 'datasets.CWRU.CWRU'>
        self.datasets = {}
        #这里的args.transfer_task[0] == [2]
        if isinstance(args.transfer_task[0],str):       #判断对象是否是str的实例，即表示字符串的对象    这里得到的应该是false 不执行if语句里的内容
           print('111',args.transfer_task)
           args.transfer_task= eval("".join(args.transfer_task))    #将args.transfer_task拼接成字符串，然后使用eval对这个字符串进行求值
        #数据划分 将数据集分割成源域训练集、源域验证集、目标域训练集和目标域验证集 用于迁移学习
        self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_train'], self.datasets['target_val'],self.datasets['target_test'] = Dataset(args.data_dir, args.transfer_task, args.normlizetype).data_split(transfer_learning=True)

        """
            数据装载 定义一个字典 其中包含四个键值对
            每个键对应一个数据集，值是一个PyTorch的DataLoader对象，用于数据集的迭代器
        """
        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False), #如果是训练集，则为True，否则为False
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),   #如果使用GPU，则为True，否则为False
                                                           drop_last=True)   #如果是训练集且参数args.last_batch为True，则为True，否则为False
                            for x in ['source_train', 'source_val', 'target_train', 'target_val','target_test']}

        # Define the model  DAGCN
        self.model = getattr(models, args.model_name)(args.pretrained)
        """
            这段代码是构建一个分类器的结构，主要包括两个部分：bottleneck_layer和classifier_layer
            bottleneck_layer于将特征向量压缩到更低的维度，以便减少计算成本和防止过拟合
            如果开启了bottleneck，输入特征向量经过一个线性层和激活函数（ReLU），并且添加一个dropout层
            再通过一个线性层将输出特征向量映射到分类的结果
        """
        if args.bottleneck:
            self.bottleneck_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),  #bottleneck_num设置为了256
                                                  nn.ReLU(inplace=True), nn.Dropout())
            self.classifier_layer = nn.Linear(args.bottleneck_num, Dataset.num_classes)     #Dataset.num_classes设置为4
        else:
            self.classifier_layer = nn.Linear(self.model.output_num(), Dataset.num_classes)

        self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)    #完整的模型
        """
            这段代码的作用是检查是否开启了域对抗（domain adversarial）的训练方式
            如果开启了域对抗，则会初始化一个AdversarialNet模型并将其作为GDDANet模型的一部分
        """
        if args.domain_adversarial:
            self.max_iter = len(self.dataloaders['source_train'])*(args.max_epoch-args.middle_epoch)
            self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=self.model.output_num(),
                                                                        hidden_size=args.hidden_size, max_iter=self.max_iter)#hidden_size设置为了1024
        #如果有多个GPU，并行计算
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            if args.bottleneck:
                self.bottleneck_layer = torch.nn.DataParallel(self.bottleneck_layer)
            if args.domain_adversarial:
                self.AdversarialNet = torch.nn.DataParallel(self.AdversarialNet)
            self.classifier_layer = torch.nn.DataParallel(self.classifier_layer)

        # Define the learning parameters 每个参数都被赋予相同的学习率args.lr 1e-3 即 0.001
        if args.domain_adversarial:
            if args.bottleneck:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},   #1e-3
                                  {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet.parameters(), "lr": args.lr}]
            else:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet.parameters(), "lr": args.lr}]
        else:
            if args.bottleneck:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr}]
            else:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr}]


        # Define the optimizer
        # 这段代码根据传入的 args.opt 参数选择使用不同的优化器来优化模型参数
        # 如果 args.opt 为 'sgd'，则使用 SGD 优化器
        # 如果 args.opt 为 'adam'，则使用 Adam 优化器
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(parameter_list, lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(parameter_list, lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        """
        这段代码是用来根据用户输入的参数选择合适的学习率调整策略
        如果 args.lr_scheduler 参数设置为 'step'，那么将会按照用户输入的步数和步长使用多步学习率调度器
        如果设置为 'exp'，则使用指数式学习率调度器
        如果设置为 'stepLR'，则使用步进学习率调度器
        如果设置为 'fix'，则不使用学习率调度器
        如果设置为 'transferLearning'，则使用迁移学习中的学习率调度器
        如果 args.lr_scheduler 参数的值不是上述任何一种情况，则会抛出异常
        """
        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]   #[150, 250]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma) #在150次和250次 lr乘0.1
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps[0:3])        #这里改了原代码不然感觉会出错 源代码是 steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        elif args.lr_scheduler == 'transferLearning':
            param_lr = []       #param_lr 是一个包含当前优化器所有参数组的学习率的列表，其中每个元素都是一个浮点数
            for param_group in self.optimizer.param_groups:
                param_lr.append(param_group["lr"])  #在循环中逐个获取每个参数组的学习率，然后添加到列表中
            self.lr_scheduler = transferLearning(self.optimizer, param_lr, args.max_epoch)
        else:
            raise Exception("lr schedule not implement")


        """
            这段代码实现了模型训练过程中的模型状态和优化器状态的恢复功能，也就是从之前保存的checkpoint中加载模型和优化器的状态，继续之前的训练
            如果 args.resume 不为空，就判断是 .tar 还是 .pth 结尾，根据文件后缀名加载不同格式的checkpoint
            如果是 .tar，则使用 torch.load 函数加载checkpoint文件中保存的模型和优化器状态，将模型和优化器的参数设置为这些状态，并且把训练开始的epoch设为checkpoint中保存的epoch加1
            如果是 .pth 文件，则使用 torch.load 函数加载模型状态并将其设置为 self.model_all 的状态
            注意， .pth 文件中没有保存优化器状态，因此需要重新创建优化器
        """
        # Load the checkpoint
        self.start_epoch = 0
        if args.resume:     #默认是''
            suffix = args.resume.rsplit('.', 1)[-1]
            if suffix == 'tar':
                checkpoint = torch.load(args.resume)
                self.model_all.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suffix == 'pth':
                self.model_all.load_state_dict(torch.load(args.resume, map_location=self.device))

        """
        这段代码是将模型及其子模块移动到指定设备上，其中 self.device 是在程序开始时根据用户指定的设备名称或者默认的 cuda 或 cpu 进行设置的
        self.model.to(self.device) 语句将整个模型移动到 self.device 设备上，这个模型是在构建模型时定义的
        如果使用了瓶颈层，则需要把瓶颈层也移动到指定设备上，这一步是通过 self.bottleneck_layer.to(self.device) 实现的
        如果使用了领域对抗训练，还需要将领域对抗网络移动到指定设备上，这一步是通过 self.AdversarialNet.to(self.device) 实现的。
        最后，分类器层通过 self.classifier_layer.to(self.device) 移动到指定设备上。
        """
        # Invert the model and define the loss
        self.model.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer.to(self.device)
        if args.domain_adversarial:
            self.AdversarialNet.to(self.device)
        self.classifier_layer.to(self.device)

        # Define the adversarial loss
        """
            在这段代码中，定义了三个不同的损失函数：
            self.adversarial_loss：定义了二分类的二元交叉熵损失函数，用于域自适应训练时判别器网络的损失计算
            self.structure_loss：定义了一种结构化的损失函数，称为DAN，用于域自适应训练时源域和目标域之间的距离度量，可以帮助域自适应训练时提高分类器的泛化能力
            self.criterion：定义了交叉熵损失函数，用于分类任务的损失计算
        """
        self.adversarial_loss = nn.BCELoss()

        self.structure_loss = DAN

        self.criterion = nn.CrossEntropyLoss()

        self.gatcriterion = nn.CrossEntropyLoss()


    def train(self):
        """可视化结果"""
        source_train_Loss = []
        source_val_Loss = []
        target_val_Loss = []
        source_train_Acc = []
        source_val_Acc = []
        target_val_Acc = []
        """
        Training process
        :return:
        """
        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()

        save_list = Save_Tool(max_num=args.max_model_num)   #保存模型文件路径
        iter_num = 0
        for epoch in range(self.start_epoch, args.max_epoch):   # 0 - 300 也就是299次epoch
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)   #在日志中记录当前训练的epoch数和最大epoch数  -----Epoch epoch/max_epoch-----
            # Update the learning rate
            """
                这段代码用于调整学习率，如果定义了学习率调度器，则会根据设定的规则来调整学习率，并打印当前的学习率
                否则直接打印初始学习率
            """
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))
            #将target domain的训练集数据封装成一个可迭代对象，方便后续的训练过程中每次从数据集中取出一个batch的数据进行训练
            iter_target = iter(self.dataloaders['target_train'])
            #得到target_train数据的长度
            len_target_loader = len(self.dataloaders['target_train'])
            # Each epoch has a training and val phase
            """对于训练过程中的不同阶段(phase)，分别遍历源域训练集(source_train)、源域验证集(source_val)、目标域验证集(target_val)"""
            for phase in ['source_train', 'source_val', 'target_val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0

                # Set model to train mode or test mode
                """
                    这段代码是针对不同的训练阶段(phase)来设置不同的模型状态
                    如果当前阶段(phase)为'source_train'，则设置模型为训练状态，包括bottleneck_layer（如果使用了bottleneck），AdversarialNet（如果使用了domain adversarial），以及classifier_layer
                    如果当前阶段(phase)为'source_val'或者'target_val'，则设置模型为评估状态，即模型不会被更新，只是用来计算性能指标
                    在这种情况下，如果使用了bottleneck和domain adversarial，则也会设置对应的模块为评估状态
                """
                if phase == 'source_train':
                    self.model.train()
                    if args.bottleneck:
                        self.bottleneck_layer.train()
                    if args.domain_adversarial:
                        self.AdversarialNet.train()
                    self.classifier_layer.train()
                else:
                    self.model.eval()
                    if args.bottleneck:
                        self.bottleneck_layer.eval()
                    if args.domain_adversarial:
                        self.AdversarialNet.eval()
                    self.classifier_layer.eval()
                """
                    使用enumerate()函数对self.dataloaders[phase]进行遍历，遍历的结果会被打包成(batch_idx, (inputs, labels))的形式
                    batch_idx表示当前遍历的批次的索引，inputs表示当前批次的输入数据，labels表示当前批次的标签数据
                """
                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    """
                        这段代码是在迭代数据集时，根据当前所处的训练阶段phase，来选择如何处理输入数据和标签
                        如果当前不是在源域训练阶段（phase != 'source_train'），或者epoch < args.middle_epoch，则直接将输入数据和标签都送到设备上进行计算
                        否则需要将源域数据和目标域数据拼接在一起作为模型的输入
                        这里通过 iter_target.__next__() 从目标域数据加载器中迭代取出一个 batch 的数据和标签
                        然后将源域数据和目标域数据按照第一个维度进行拼接，并将它们都送到设备上进行计算
                    """
                    if phase != 'source_train' or epoch < args.middle_epoch:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    else:
                        source_inputs = inputs
                        target_inputs, target_labels = iter_target.__next__()
                        inputs = torch.cat((source_inputs, target_inputs), dim=0)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    """
                        这行代码的作用是当在训练过程中遍历完一轮target数据集之后，重新初始化一个iterator来遍历target数据集，确保下一轮遍历时能够从数据集的开头开始
                        这样做是因为在PyTorch中，一个iterator遍历完数据集之后不能继续遍历，需要重新创建一个iterator才能再次遍历
                    """
                    if (step + 1) % len_target_loader == 0:
                        iter_target = iter(self.dataloaders['target_train'])

                    with torch.set_grad_enabled(phase == 'source_train'):   #训练阶段，张量的梯度开启，其他阶段关闭
                        # forward   前向传播
                        """
                        这段代码是输入inputs到模型中，然后将输出结果作为features
                        并且在参数设置中有一个args.bottleneck，如果它为真，则会将features输入到瓶颈层self.bottleneck_layer中，然后将瓶颈层的输出作为最终输出
                        再通过分类器层self.classifier_layer输出预测结果outputs
                        """
                        features = self.model(inputs)

                        if args.bottleneck:
                            features = self.bottleneck_layer(features)
                        outputs = self.classifier_layer(features)
                        """
                            在源域和目标域共享标签的训练方式中，在源域上正常训练分类器
                            但是在目标域上，分类器的输出在第一次输出后被截断，以避免使用目标域的标签信息进行反向传播和更新，从而防止源域特征的丢失
                            因此在这种情况下使用分类器的输出进行计算损失的方式不同，需要将其截断后再计算分类器的损失
                        """
                        if phase != 'source_train' or epoch < args.middle_epoch:
                            logits = outputs
                            loss = self.criterion(logits, labels)


                        else:
                            logits = outputs.narrow(0, 0, labels.size(0))   #这里的长度是只含有训练数据标签的长度
                            classifier_loss = self.criterion(logits, labels)

                            #---------------------------------------------------
                            """！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！"""
                        if phase == 'source_train' and epoch >= args.middle_epoch:
                            # Calculate the domain adversarial

                            domain_label_source = torch.ones(labels.size(0)).float()    #全是1
                            domain_label_target = torch.zeros(inputs.size(0)-labels.size(0)).float()#全是0  输入是按行cat起来的 前面是源域input 后面是target input
                            adversarial_label = torch.cat((domain_label_source, domain_label_target), dim=0).to(self.device)
                            adversarial_label = adversarial_label.unsqueeze(1)
                            """
                            这个方法会执行 AdversarialNet 类的 forward 方法
                            这是因为在 AdversarialNet 类中重载了 nn.Module 类的 forward 方法
                            所以在调用 self.AdversarialNet(features) 时，实际上是在调用 AdversarialNet 类的 forward 方法
                            """




                            adversarial_out = self.AdversarialNet(features)
                            #adversarial_out = self.AdversarialNet.forward(features)
                            adversarial_loss = self.adversarial_loss(adversarial_out, adversarial_label)
                            #从 features 中取出源域数据和目标域数据，然后分别计算源域和目标域数据之间的结构损失
                            structure_loss = self.structure_loss(features.narrow(0, 0, labels.size(0)),
                                                               features.narrow(0, labels.size(0),inputs.size(0) - labels.size(0)))

                            if args.trade_off_adversarial == 'Cons':
                                lam_adversarial = args.lam_adversarial      #1
                                """
                                    lam_adversarial 的值会随着训练 epoch 的增加而变化
                                    具体来说，当epoch小于args.middle_epoch时，lam_adversarial 的值等于-1
                                    当epoch大于等于args.middle_epoch时，lam_adversarial 的值等于一个在 [-1, 1] 之间的 sigmoid 函数值
                                """
                            elif args.trade_off_adversarial == 'Step':
                                """这个值表示训练从开始到当前所占的比例，因此这个值可以控制lam_adversarial随着训练的进展而改变，从而达到更好的训练效果"""
                                lam_adversarial = 2 / (1 + math.exp(-10 * ((epoch-args.middle_epoch) /
                                                                        (args.max_epoch-args.middle_epoch)))) - 1
                            else:
                                raise Exception("loss not implement")

                            """
                                这行代码的作用是计算总的损失值，它是分类损失、领域对抗损失、领域结构损失三者之和
                                lam_adversarial是控制领域对抗和领域结构损失的权重系数
                            """
                            loss = classifier_loss  + lam_adversarial * adversarial_loss + lam_adversarial * structure_loss
                            

                        pred = logits.argmax(dim=1) #logits张量包含每个类别的得分，argmax(dim=1)的返回值是在第1维度上最大值的下标，即预测的类别
                        """
                            用于比较模型的预测pred和实际标签labels是否相等，生成一个布尔类型的张量
                            接着，使用.float()方法将布尔类型的张量转换成浮点类型的张量，方便后续计算
                            使用.sum()方法计算所有预测正确的样本数量，再使用.item()方法将其转换成Python标量返回
                            最终，correct保存了预测正确的样本数量
                        """
                        correct = torch.eq(pred, labels).float().sum().item()

                        """
                            这行代码是将每个样本的loss乘以该batch中的样本数量，即将 batch中所有样本的loss求和
                            它的目的是为了最后能够得到整个epoch的平均loss
                            loss.item() 得到的是一个标量值，也就是一个张量中的单个值
                            这个值通常代表了损失函数的大小，可以用来衡量模型的性能
                            这个值乘上labels.size(0)以后就得到了当前batch中的总损失值
                        """
                        loss_temp = loss.item() * labels.size(0)
                        #epoch_loss 是训练过程中所有 batch 的 loss 的和
                        epoch_loss += loss_temp
                        """
                            epoch_acc是一个累计变量，用来记录一个epoch中分类器预测正确的样本数量
                            每次计算损失和预测结果后，通过计算当前批次中分类器预测正确的样本数量correct，将其累加到epoch_acc中
                            最后在该epoch结束后，epoch_acc的值除以数据集中的样本总数即可得到该epoch中分类器的准确率
                        """
                        epoch_acc += correct

                        """
                            epoch_length是每个epoch中的样本总数，等于所有batch中的样本数量之和
                            在每个batch中，labels.size(0)表示当前batch中的样本数，所以将其加入到epoch_length中可以得到该epoch中的样本总数
                        """
                        epoch_length += labels.size(0)

                        # Calculate the training information
                        """这段代码是模型训练中最为重要的部分，主要包含以下几个步骤"""
                        if phase == 'source_train':
                            # backward
                            """在源域训练中，先将梯度清零，然后执行反向传播算法计算损失函数关于参数的梯度，最后利用优化器来更新参数"""
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            """记录当前batch的损失值、正确率以及样本数，并在一定的训练批次之后打印训练信息"""
                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += labels.size(0)
                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx * len(labels), len(self.dataloaders[phase].dataset),
                                    batch_loss, batch_acc, sample_per_sec, batch_time
                                ))
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            """更新当前批次的索引号，以便下一次迭代使用"""
                            step += 1
                            """通过这些步骤，可以完成整个模型的训练过程，每个批次的损失和正确率都会被计算和记录，同时也会输出当前的训练信息"""

                # Print the train and val information via each epoch
                """
                    这部分的代码用于计算每个epoch的平均损失（loss）和准确率（accuracy）
                    epoch_length为该epoch中样本的总数，epoch_loss和epoch_acc初始化为0，分别用于计算该epoch的总损失和总准确率
                    最后通过将总损失和总准确率除以epoch中样本的总数，计算出该epoch的平均损失和平均准确率
                """
                epoch_loss = epoch_loss / epoch_length
                epoch_acc = epoch_acc / epoch_length
                #整个字符串的意思是输出当前epoch的相关信息，包括损失、准确率和运行时间，用于监控模型的训练情况
                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))
                if phase == 'source_train':
                  source_train_Loss.append(epoch_loss)
                  source_train_Acc.append(epoch_acc)
                if phase == 'source_val':
                  source_val_Loss.append(epoch_loss)
                  source_val_Acc.append(epoch_acc)
                if phase == 'target_val':
                  target_val_Loss.append(epoch_loss)
                  target_val_Acc.append(epoch_acc)
    
                # save the model
                """
                    这部分代码实现的功能是：在目标域的验证集上，每个epoch保存一次模型checkpoint，并根据验证集准确率的表现保存最好的模型
                """
                if phase == 'target_val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model_all.state_dict()
                    save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(epoch))
                    torch.save({
                        'epoch': epoch,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'model_state_dict': model_state_dic
                    }, save_path)
                    save_list.update(save_path)
                    # save the best model according to the val accuracy
                    if (epoch_acc > best_acc or epoch > args.max_epoch-2) and (epoch > args.middle_epoch-1):
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                    os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))
        plt.figure()
        plt.plot(range(1, len(source_train_Loss) + 1), source_train_Loss, label='source_train_Loss')
        plt.plot(range(1, len(source_val_Loss) + 1), source_val_Loss, label='source_val_Loss')
        plt.plot(range(1, len(target_val_Loss) + 1), target_val_Loss, label='target_val_Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(args.loss_png)

        plt.figure()
        plt.plot(range(1, len(source_train_Acc) + 1), source_train_Acc, label='source_train_Acc')
        plt.plot(range(1, len(source_val_Acc) + 1), source_val_Acc, label='source_val_Acc')
        plt.plot(range(1, len(target_val_Acc) + 1), target_val_Acc, label='target_val_Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend()
        plt.savefig(args.acc_png)




    #在目标域上进行测试
    def test(self):
        args = self.args

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count" #抛出GPU分配异常
        else:
            warnings.warn("gpu is not available")       #没有GPU，转而使用CPU
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # Load the datasets     加载数据集
        Dataset = getattr(datasets, args.data_name)  # 得到的是<class 'datasets.CWRU.CWRU'>


        # 创建模型实例
        # Define the model  DAGCN
        self.model = getattr(models, args.model_name)(args.pretrained)
        """
            这段代码是构建一个分类器的结构，主要包括两个部分：bottleneck_layer和classifier_layer
            bottleneck_layer于将特征向量压缩到更低的维度，以便减少计算成本和防止过拟合
            如果开启了bottleneck，输入特征向量经过一个线性层和激活函数（ReLU），并且添加一个dropout层
            再通过一个线性层将输出特征向量映射到分类的结果
        """
        if args.bottleneck:
            self.bottleneck_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),
                                                  # DAGCN.output_num设置为256 bottleneck_num设置为了256
                                                  nn.ReLU(inplace=True), nn.Dropout())
            self.classifier_layer = nn.Linear(args.bottleneck_num, Dataset.num_classes)  # Dataset.num_classes设置为10
        else:
            self.classifier_layer = nn.Linear(self.model.output_num(), Dataset.num_classes)

        self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)  # 完整的模型

        if args.domain_adversarial:
            self.max_iter = len(self.dataloaders)*(args.max_epoch-args.middle_epoch)
            self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=self.model.output_num(),
                                                                        hidden_size=args.hidden_size, max_iter=self.max_iter)#hidden_size设置为了1024

        # Invert the model and define the loss
        self.model.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer.to(self.device)
        if args.domain_adversarial:
            self.AdversarialNet.to(self.device)
        self.classifier_layer.to(self.device)

        #加载模型参数
        self.model_all.load_state_dict(torch.load(args.param_pth))

        #设置模型为评估模式：在进行模型测试之前，确保将模型设置为评估模式。这可以通过调用模型的eval()方法来实现，以确保模型在测试阶段不会进行梯度计算和参数更新
        self.model_all.eval()

        epoch_acc = 0

        epoch_length = 0

        for inputs,labels in self.dataloaders['target_test']:
            """
            inputs torch.Size([64, 1, 1024])
            labels torch.Size([64])

            print('inputs',inupts.shape)
            print('labels',labels.shape)
            """
            inputs = inputs.type(torch.cuda.FloatTensor)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)


            features = self.model(inputs)  # 经过DAGCN特征提取 得到的是代码固定的256维的特征向量 [batch_size,256]
            if args.bottleneck:
                features = self.bottleneck_layer(features)  # 通过瓶颈结构 输入维度 * 256 的线性层 + ReLu激活层 + Dropout
            outputs = self.classifier_layer(features)

            pred = outputs.argmax(dim=1)

            correct = torch.eq(pred, labels).float().sum().item()

            epoch_acc += correct

            epoch_length += labels.size(0)



        epoch_acc = epoch_acc / epoch_length
        logging.info('{:.4f}'.format(
              epoch_acc
        ))
        print('epoch_acc',epoch_acc)

    def t_SNE_s_t(self):
        # 修改全局字体大小
        plt.rc('font', size=14)  # 设置字体大小
        plt.rc('axes', labelsize=14)  # 设置坐标轴标签字体大小
        plt.rc('legend', fontsize=12)  # 设置图例字体大小
        args = self.args

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        Dataset = getattr(datasets, args.data_name)

        # Rest of the code ...

        # Load source train data
        # source_features = []
        # source_labels_list = []
        # for inputs, labels in self.dataloaders['source_val']:
        #     inputs = inputs.to(self.device)
        #     labels = labels.to(self.device)
        #     hidden_layer_features = self.model(inputs)
        #     hidden_layer_features = hidden_layer_features.to('cpu').detach().numpy()
        #     source_features.append(hidden_layer_features)
        #     source_labels_list.append(labels)
        # source_features = np.concatenate(source_features, axis=0)
        # source_labels = torch.cat(source_labels_list, dim=0).cpu().numpy()
        #
        # # t-SNE for source train data
        # source_tsne = TSNE(n_components=2, learning_rate=200, perplexity=15, n_iter=1000)
        # source_X_tsne = source_tsne.fit_transform(source_features)

        # Load target test data
        target_features = []
        target_labels_list = []
        for inputs, labels in self.dataloaders['target_test']:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            hidden_layer_features = self.model(inputs)
            hidden_layer_features = hidden_layer_features.to('cpu').detach().numpy()
            target_features.append(hidden_layer_features)
            target_labels_list.append(labels)
        target_features = np.concatenate(target_features, axis=0)
        target_labels = torch.cat(target_labels_list, dim=0).cpu().numpy()

        # t-SNE for target test data
        target_tsne = TSNE(n_components=2, learning_rate=200, perplexity=15, n_iter=1000)
        target_X_tsne = target_tsne.fit_transform(target_features)

        # Plot both source and target data on the same plot with the same color for the same labels
        # unique_labels = np.unique(np.concatenate((source_labels, target_labels)))

        # Define a colormap for different labels
        # Define a colormap for 20 different labels
        cmap = plt.cm.get_cmap('tab10', 10)

        plt.figure(figsize=(10, 8))  # Set the figure size if needed

        for label in range(10):
            # source_mask = source_labels == label
            target_mask = target_labels == label

            # Get the color from the colormap
            color = cmap(label)

            # plt.scatter(source_X_tsne[source_mask, 0], source_X_tsne[source_mask, 1], c=[color], marker='o', alpha=1,
            #             label='S{}'.format(label))
            plt.scatter(target_X_tsne[target_mask, 0], target_X_tsne[target_mask, 1], c=[color], marker='x', alpha=1,
                        label='T{}'.format(label),s=100)

        # Add legend outside of the plot area
        # plt.legend(prop={'size': 5}, bbox_to_anchor=(1.05, 1), loc='upper center')
        # 添加图例

        # 设置图例的位置，使其位于正上方
        # legend = plt.legend(prop={'size': 14}, bbox_to_anchor=(0.5, 1.15), loc='upper center')
        # 设置图例的位置，使其位于正上方，同时水平排列
        legend = plt.legend(prop={'size': 14}, bbox_to_anchor=(0.5, 1.25), loc='upper center', ncol=5)

        # 修改图例标签的大小
        for label in legend.get_texts():
            label.set_fontsize(14)  # 这里设置图例标签的字体大小

        # plt.subplots_adjust(right=0.8)  # Adjust the layout to make space for the legend

        plt.savefig('pr16.png', dpi=700, bbox_inches='tight')
        plt.close()

    def t_SNE(self):
        args = self.args

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count" #抛出GPU分配异常
        else:
            warnings.warn("gpu is not available")       #没有GPU，转而使用CPU
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # Load the datasets     加载数据集
        Dataset = getattr(datasets, args.data_name)  # 得到的是<class 'datasets.CWRU.CWRU'>

        self.model = getattr(models, args.model_name)(args.pretrained)


        if args.bottleneck:
            self.bottleneck_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),
                                                  # DAGCN.output_num设置为256 bottleneck_num设置为了256
                                                  nn.ReLU(inplace=True), nn.Dropout())
            self.classifier_layer = nn.Linear(args.bottleneck_num, Dataset.num_classes)  # Dataset.num_classes设置为10
        else:
            self.classifier_layer = nn.Linear(self.model.output_num(), Dataset.num_classes)

        self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)  # 完整的模型

        if args.domain_adversarial:
            self.max_iter = len(self.dataloaders)*(args.max_epoch-args.middle_epoch)
            self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=self.model.output_num(),
                                                                        hidden_size=args.hidden_size, max_iter=self.max_iter)#hidden_size设置为了1024

        # # Invert the model and define the loss
        self.model.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer.to(self.device)
        if args.domain_adversarial:
            self.AdversarialNet.to(self.device)
        self.classifier_layer.to(self.device)

        # 加载模型参数
        self.model_all.load_state_dict(torch.load(args.param_pth))


        self.model_all.eval()


        features = []
        labels_list = []

        # 加载测试集数据
        for  inputs,labels in self.dataloaders['target_test']:

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            hidden_layer_features = self.model(inputs)

            hidden_layer_features = hidden_layer_features.to('cpu').detach().numpy()

            features.append(hidden_layer_features)
            labels_list.append(labels)

        features = np.concatenate(features, axis=0)
        labels = torch.cat(labels_list, dim=0).cpu().numpy()
        # 使用t-SNE进行降维
        tsne = TSNE(n_components=2, learning_rate=200, perplexity=30, n_iter=1000)
        X_tsne = tsne.fit_transform(features)


        # 可视化结果（区分不同类别的颜色）

        plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=labels)

        plt.show()

        # plt.savefig('t-sne.png')
        #
        # # 关闭绘图窗口
        # plt.close()

    def vis(self):
        args = self.args

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"  # 抛出GPU分配异常
        else:
            warnings.warn("gpu is not available")  # 没有GPU，转而使用CPU
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # Load the datasets     加载数据集
        Dataset = getattr(datasets, args.data_name)  # 得到的是<class 'datasets.CWRU.CWRU'>

        self.model = getattr(models, args.model_name)(args.pretrained)

        if args.bottleneck:
            self.bottleneck_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),
                                                  # DAGCN.output_num设置为256 bottleneck_num设置为了256
                                                  nn.ReLU(inplace=True), nn.Dropout())
            self.classifier_layer = nn.Linear(args.bottleneck_num, Dataset.num_classes)  # Dataset.num_classes设置为10
        else:
            self.classifier_layer = nn.Linear(self.model.output_num(), Dataset.num_classes)

        self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)  # 完整的模型

        if args.domain_adversarial:
            self.max_iter = len(self.dataloaders) * (args.max_epoch - args.middle_epoch)
            self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=self.model.output_num(),
                                                                    hidden_size=args.hidden_size,
                                                                    max_iter=self.max_iter)  # hidden_size设置为了1024

        # # Invert the model and define the loss
        self.model.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer.to(self.device)
        if args.domain_adversarial:
            self.AdversarialNet.to(self.device)
        self.classifier_layer.to(self.device)

        # 加载模型参数
        self.model_all.load_state_dict(torch.load(args.param_pth))

        self.model_all.eval()

        # 选择一个示例输入样本
        sample_inputs, sample_labels = next(iter(self.dataloaders['target_test']))
        sample_inputs = sample_inputs.type(torch.cuda.FloatTensor)
        sample_inputs = sample_inputs.to(self.device)
        sample_labels = sample_labels.to(self.device)

        # 前向传播
        features = self.model(sample_inputs)  # 经过DAGCN特征提取，得到固定的256维特征向量 [batch_size, 256]
        if args.bottleneck:
            features = self.bottleneck_layer(features)  # 通过瓶颈结构 输入维度 * 256 的线性层 + ReLu激活层 + Dropout
        outputs = self.classifier_layer(features)

        # 可视化模型
        dot = make_dot(outputs, params=dict(self.model.named_parameters()))
        dot.render(filename='model_graph', format='png')







          

















