#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019-04-13 16:40
#@Author  :Wanuix 
#@FileName: 项目_01_数据处理01.py

#@Software: PyCharm

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv

# ============================数据预处理================================== #
print(os.getcwd())  # $-pwd
os.chdir('/Users/shan/Documents/Python学习/城市公交站点设置的优化分析/数据')
# ********************************************************************** #
# -------   读取数据    -------- #
# 方法一：
data_gps_09 = pd.read_csv('./gps/gps_20140609.csv', sep=',', encoding='gbk')
print(data_gps_09.head())
# 方法二：
# f = open('./gps/gps_20140609.csv', mode='r+', encoding='gbk')
# a = f.read()
# data = csv.reader(f)
# data_gps_09 = [shuju for shuju in data]
# data_gps_09 = pd.DataFrame(data_gps_09)
# print(data_gps_09)
# f.close()
# # 方法三：
# try:
#     f = open('./gps/gps_20140609.csv', mode='r+', encoding='gbk')
#     a = f.read()
#     data = csv.reader(f)
#     data_gps_09 = [shuju for shuju in data]
#     data_gps_09 = pd.DataFrame(data_gps_09)
# finally:
#     if f:
#         f.close()
# # 方法四：
# with open('./gps/gps_20140609.csv', mode='r+', encoding='gbk') as f:
#     data = csv.reader(f)
#     data_gps_09 = [shuju for shuju in data]
#     data_gps_09 = pd.DataFrame(data_gps_09)
# ********************************************************************** #
# -------   数据清洗及处理    -------- #
# pandas中的空值处理 https://www.cnblogs.com/louyifei0824/p/9942430.html
# 检查空值、缺失值
print('-'*100)
if data_gps_09.notnull().shape == data_gps_09.shape:
    print("无空缺值")
    print('-'*100)
    print("[属性]\t\t[是否为空值]")
    print(data_gps_09.isnull().any())
else:
    print("存在空缺值,个数为：%d个" %
          list(data_gps_09.shape)[0]-list(data_gps_09.notnull().shape)[0])
# 过滤丢失数据
data_gps_09.dropna()

# 数据去重 df.duplicated()
# 本项目基于业务时间、卡片记录编码、车牌号，对数据去重
print('-'*100)
data_gps_09_isduplicated = \
    data_gps_09.duplicated(['业务时间', '卡片记录编码', '车牌号'], keep='first')
temp = 0
for i in range(data_gps_09_isduplicated.shape[0]):
    if data_gps_09_isduplicated[i]:
        temp = 1
        break
del data_gps_09_isduplicated
if temp == 0:
    print("基于业务时间、卡片记录编码、车牌号属性查询，无重复数据，不需要去重")
else:
    print("基于业务时间、卡片记录编码、车牌号属性查询，有重复数据，需要进行数据去重")
    data_gps_09 = data_gps_09.drop_duplicates(
        ['业务时间', '卡片记录编码', '车牌号'], keep='first')

# 基于项目需求，只分析68路公交车，则只需要提取68路公交车的数据
data_68_gps_09 = data_gps_09.loc[data_gps_09.iloc[:, 4] == '68路', :]