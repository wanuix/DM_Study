#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019-04-15 17:55
#@Author  :Wanuix 
#@FileName: 项目_05_分时段处理.py

#@Software: PyCharm

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import csv

# ============================数据探索================================== #
os.chdir('/Users/shan/Documents/Python学习/城市公交站点设置的优化分析/数据')
sns.set()
# ********************************************************************* #
# -------   分时段处理    -------- #
df_data = pd.read_csv("gjc_zd_actual.csv", sep=',', encoding='gbk')

# 分割日期和时间，按空格号分开
T = [df_data.iloc[i, 2].split(" ") for i in list(df_data.index)]
# 提取时间,并将日期赋予同一个值2014/06/09，方便分时段
# T是列表，time[i] = T[[i]][2]表示T中第i个列表的第三列赋值给time的第i个

at = []
# for i in df_data.index:
#     time = [T[i][0], T[i][1]]
#     t = ' '.join(time)        ##合并公式
#     at.append(t)
# 或者用列表推导式的方法
at = [' '.join([T[i][0], T[i][1]]) for i in df_data.index]

time1 = [time.strptime(i, '%Y/%m/%d %H:%M') for i in at]
time2 = [time.strftime('%Y-%m-%d %H:%M', j) for j in time1]
df_data['业务时间'] = time2

# 分时段提取数据
# 设置时间点
point = ["2014/06/09 05:00",
         "2014/06/09 08:00",
         "2014/06/09 09:00",
         "2014/06/09 18:00",
         "2014/06/09 19:00",
         "2014/06/09 23:59"]
time3 = [time.strptime(i, '%Y/%m/%d %H:%M') for i in point]
time4 = [time.strftime('%Y-%m-%d %H:%M', j) for j in time3]

# 设置写出路径
lj = ["./2014-06-09-分段数据提取/时段1_68.csv",
      "./2014-06-09-分段数据提取/时段2_68.csv",
      "./2014-06-09-分段数据提取/时段3_68.csv",
      "./2014-06-09-分段数据提取/时段4_68.csv",
      "./2014-06-09-分段数据提取/时段5_68.csv"]
for k in range(0, len(lj)):
    kk = (df_data['业务时间'] >= time4[k]) & (df_data['业务时间'] < time4[k+1])
    gjc = df_data.loc[kk==True, :]
    gjc.to_csv(lj[k], na_rep='NaN', header=True, index=False)
