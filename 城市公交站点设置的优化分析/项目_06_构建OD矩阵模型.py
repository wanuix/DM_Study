#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019-04-15 18:34
#@Author  :Wanuix 
#@FileName: 项目_06_构建OD矩阵模型.py

#@Software: PyCharm

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
import csv
import 项目_04_密度聚类DEScan as cluster

# ============================数据探索================================== #
os.chdir('/Users/shan/Documents/Python学习/城市公交站点设置的优化分析/数据')
sns.set()
# ********************************************************************* #
# -------   构建OD矩阵模型    -------- #

####------------------------------计算上车人数----------------------------####
# df_get_on_num = cluster.df_cluster_result.groupby('cluster').count().iloc[:, 1]
# dataframe和Series没有计数函数，转化为列表太麻烦
df_get_on_num = cluster.df_cluster_result.groupby('cluster').count().iloc[:, 1]
df_get_on_num.name = ('get_on_num')     # 改名

df_flag = cluster.df_station['cluster']
df_flag.index = df_flag
df_result = pd.concat([df_flag, df_get_on_num], axis=1)

####-------------------------------吸引权重-------------------------------####
df_wj = df_result['get_on_num']/sum(df_result['get_on_num'])
df_wj.name = ('wj')
df_result = pd.concat([df_result, df_wj], axis=1)

####-------------------------------下车人数-------------------------------####
'''
k = 0
for i in range(38):
    k = k + i +1
k/38
'''
# 构建泊松分布的函数
lmd = 19.5  # 居民公交出行途径站数的数学期望 (1+2+3+..+38)/38
k = len(df_result)
pro = pd.DataFrame(np.zeros((k, k+1)))
for i in range(k):
    for j in range(k):
        if i < j:
            f = (math.e**(-lmd) * lmd**(j-i)) / math.factorial(j-i)
            pro.iloc[i, j] = f
    pro.iloc[i, k] = sum(pro.iloc[i, :k] * df_wj)
    # 这里最好不要用ix，因为行名，列名都为1：:39数字，会混淆。最好用iloc位置索引

####------------------------------OD矩阵----------------------------------####
# 构建OD矩阵,求出一个站点到另一个站点的下车人数
# 创建数据框
df_OD = pd.DataFrame(np.zeros((k+1, k+1)))
for i in range(k):
    for j in range(k):
        if i < j:
            p = pro.iloc[i, j]*df_wj.iloc[j] / pro.iloc[i, k]
            df_OD.iloc[i, j] = round(p * df_result.iloc[i, 1])

# 求出OD数据框每列的人数的总和，即为每个站点下车的总人数
list_get_off_num = list()
for j in range(k):
    list_get_off_num.append(sum(df_OD.iloc[:k, j]))
df_get_off_num = pd.DataFrame(list_get_off_num, columns=['get_off_num'], index=df_result.index)

df_result = pd.concat([df_result, df_get_off_num], axis=1)


# 各站点下车人数
for i in range(k):
    df_OD.iloc[k, i] = sum(df_OD.iloc[:k, i])
# 各站点上车人数
for i in range(k):
    df_OD.iloc[i, k] = sum(df_OD.iloc[i, :k])

# 上车总人数
sum(df_OD.iloc[:k, k])
# 下车总人数
sum(df_OD.iloc[k, :k])

df_OD.iloc[k, k] = sum(df_OD.iloc[k, :k])

# 保存结果
df_OD.to_csv('./68_all_OD.csv')

df_OD_01 = df_OD.iloc[:,:-1]
df_OD_02 = df_OD_01.iloc[:-1,:]
df_OD_02.to_csv('./68_all_OD_Matrix.csv', na_rep='NaN', header=False, index=False)

plt.figure(figsize=(8, 4))
cax = plt.matshow(df_OD_02, vmin=0, vmax=25)
plt.colorbar(cax)
ticks = np.arange(0, 38, 2)
plt.xticks(ticks)
plt.yticks(ticks)
plt.xlabel('下车')
plt.ylabel('上车')
plt.show()

# from pandas.tools.plotting import scatter_matrix
# scatter_matrix(df_OD_02,figsize=(10, 10))
# plt.show()
#
# plt.figure(figsize=(8, 4))
# sns.pairplot(df_OD_02)
# sns.plt.show()
#
# plt.figure(figsize=(8, 4))
# sns.heatmap(df_OD_02, annot=True, vmax=1, square=True, cmap="Blues")
# plt.show()
# plt.figure(figsize=(8, 4))
# cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
# sns.heatmap(df_OD_02, linewidths=0.05, cmap=cmap, center=None, robust=False)