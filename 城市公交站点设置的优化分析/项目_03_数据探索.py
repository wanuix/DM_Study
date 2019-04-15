#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019-04-15 15:47
#@Author  :Wanuix 
#@FileName: 项目_03_数据探索.py

#@Software: PyCharm

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================数据探索================================== #
os.chdir('/Users/shan/Documents/Python学习/城市公交站点设置的优化分析/数据')
# ********************************************************************* #
# -------   python画图    -------- #
sns.set()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(8, 4))
file_list = os.listdir('./time')
for file_name in file_list:
    df_time = pd.read_csv('./time/' + file_name, sep=',', encoding='utf-8')
    plt.plot(df_time['date'], df_time['num'], label=file_name)

plt.xticks([5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
           ['5:00', '7:00', '9:00', '11:00', '13:00', '15:00', '17:00', '19:00', '21:00', '23:00'])
plt.yticks([0, 100000, 200000, 300000, 400000, 500000, 600000],
           ['0', '10', '20', '30', '40', '50', '60'])
plt.xlabel("时间")
plt.ylabel("刷卡人数（单位：万人）")
plt.title("五天的当日刷卡人数与时间节点关系图")
plt.legend(('2014-06-09', '2014-06-10', '2014-06-11', '2014-06-12', '2014-06-13'))
plt.show()

df_data = pd.read_csv("gjc.csv", sep=',', encoding='gbk')
plt.figure(figsize=(8, 4))
plt.scatter(df_data["经度"], df_data["纬度"])
plt.show()