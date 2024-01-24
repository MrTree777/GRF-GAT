import time
import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_utils_combines import train_utils
import torch
import warnings
# print(torch.__version__)
warnings.filterwarnings('ignore')

args = None

def parse_args():
    #创建一个解析对象
    parser = argparse.ArgumentParser(description='Train')
    # model and data parameters
    parser.add_argument('--model_name', type=str, default='GRF-GAT', help='the name of the model')
    # parser.add_argument('--data_name', type=str, default='JU', help='the name of the data')
    parser.add_argument('--data_name', type=str, default='CWRU', help='the name of the data')
    # parser.add_argument('--data_name', type=str, default='SQ', help='the name of the data')
    # parser.add_argument('--data_dir', type=str, default="E:\PythonProject\DataSets\JU_DataSet", help='the directory of the data')
    parser.add_argument('--data_dir', type=str, default="E:\PythonProject\Datasets\CW_DataSet",help='the directory of the data')
    # parser.add_argument('--data_dir', type=str, default="E:\PythonProject\DataSets\SQ_DataSet",help='the directory of the data')
    # #迁移学习任务 默认是一个二维列表 包含两个子列表[2] [1] 子列表中又包含一个整数元素
    parser.add_argument('--transfer_task', type=list, default=[[2], [3]], help='transfer learning tasks')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')

    # training parameters
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    parser.add_argument('--bottleneck', type=bool, default=True, help='whether using the bottleneck layer')
    parser.add_argument('--bottleneck_num', type=int, default=256, help='whether using the bottleneck layer')
    parser.add_argument('--last_batch', type=bool, default=True, help='whether using the last batch')

    #
    parser.add_argument('--domain_adversarial', type=bool, default=True, help='whether use domain_adversarial')
    parser.add_argument('--hidden_size', type=int, default=1024, help='whether using the last batch')
    # parser.add_argument('--trade_off_adversarial', type=str, default='Cons', help='')
    parser.add_argument('--trade_off_adversarial', type=str, default='Step', help='')
    parser.add_argument('--lam_adversarial', type=float, default=1, help='this is used for Cons')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')   #优化器
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')                #权重衰退
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step', help='the learning rate schedule')    #学习率调度
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='150,250,350,450,550,650,750,850,950', help='the learning rate decay for step and stepLR') #1

    # save, load and display information
    parser.add_argument('--resume', type=str, default='', help='the directory of the resume training model')    #恢复训练模型的路径
    parser.add_argument('--max_model_num', type=int, default=1, help='the number of most recent models to save')
    parser.add_argument('--middle_epoch', type=int, default=1, help='min number of epoch')
    parser.add_argument('--max_epoch', type=int, default=300, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=50, help='the interval of log training information')  #每50次打印信息

    #参数文件名称，图像名称
    parser.add_argument('--param_pth',type=str,default='12-0.8066-best_model.pth',help='the directory of the model param')
    parser.add_argument('--loss_png',type=str,default='Loss', help='the name of the loss picture')
    parser.add_argument('--acc_png',type=str, default='Acc', help='the name of the acc picture')
    args = parser.parse_args()      #调用方法进行解析，解析成功后即可使
    return args


if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()   #该设置只运行PyTorch使用当前编号的GPU
    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')  #子路径
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)                               #存储路径
    if not os.path.exists(save_dir):                                                    #不存在此路径，重新创建该路径
        os.makedirs(save_dir)
    # set the logger                        保存日志 在这个路径下
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args         遍历这个解析对象
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))     #将数据插入到格式化字符串中

    trainer = train_utils(args, save_dir)
    trainer.setup()


    # istrain = True
    istrain = False
    if istrain:
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        execution_time = end_time - start_time  # 计算执行时间
        print(f"执行时间：{execution_time}秒")
    else:
        trainer.test()

    #特征可视化
    trainer.t_SNE_s_t()
    #






