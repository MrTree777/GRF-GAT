from torch import nn
import numpy as np

"""
    这个函数用来计算一个系数，用于域自适应方法中对域分类器损失和对抗损失的加权
    首先计算一个中间值 alpha * iter_num / max_iter
    然后通过 sigmoid 函数将其映射到 [0, 1] 范围内
    最后根据 high 和 low 计算出系数值
    该函数计算了一个时间步上的系数值，用于调整梯度翻转层的权重，使其能够平滑地从初始状态变为目标状态
"""
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    # return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


"""
    这个函数定义了一个钩子函数 grl_hook，它用于在反向传播中给梯度乘上一个系数 coeff，以实现域对抗训练中的梯度反转
    具体地，将这个钩子函数应用到网络的某个中间层时，会在这个层的梯度计算中使用该函数，将梯度乘上 coeff 的负数，即将梯度取反，并返回新的梯度值
    在域对抗训练中，当网络在源域上训练时，这个系数设为 1，从而保留梯度信息；当网络在目标域上训练时，这个系数设为 -1，将梯度反转后再传回网络
"""
def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1

"""
    这是一个域对抗网络的定义，其中包含了三个线性层和一个sigmoid激活函数
    这个网络的目的是通过域自适应学习一个分类器，以使得在不同的域中分类器的性能能够更加鲁棒
    在forward函数中，通过计算coeff来得到反向传播时的梯度系数，并将输入x和coeff传入grl_hook函数中
    grl_hook函数通过将输入梯度乘以一个系数的负数来实现梯度反向传播的变换，最终输出网络的预测结果y。这个网络中还定义了一个output_num函数，用于返回模型输出的维度
"""
class AdversarialNet(nn.Module):
    def __init__(self, in_feature, hidden_size,max_iter = 10000.0):
        super(AdversarialNet, self).__init__()
        self.ad_layer1 = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.ad_layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        # parameters
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter
        self.__in_features = 1

    def forward(self, x):

        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))

        x = self.ad_layer1(x)
        x = self.ad_layer2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return self.__in_features
