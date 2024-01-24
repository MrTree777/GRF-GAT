#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging

def setlogger(path):
    logger = logging.getLogger()        #获取一个logger对象
    logger.setLevel(logging.INFO)       #设置日志级别为'INFO'
    logFormatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")   #创建一个Formatter对象，并设置输出格式为时间、消息

    fileHandler = logging.FileHandler(path) #创建一个FileHandler对象，设置输出文件名名——   ./checkpoint/DAGCN_features_0510-081922/train.log
    fileHandler.setFormatter(logFormatter)  #设置输出日志的格式为logFormatter指定的格式
    logger.addHandler(fileHandler)          #将FileHandler对象添加到logger对象中

    consoleHandler = logging.StreamHandler()    #创建一个StreamHandler对象
    consoleHandler.setFormatter(logFormatter)   #设置输出格式
    logger.addHandler(consoleHandler)           #将Handler对象添加到logger对象中

