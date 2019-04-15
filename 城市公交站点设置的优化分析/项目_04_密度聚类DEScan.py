#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019-04-15 17:24
#@Author  :Wanuix 
#@FileName: 项目_04_密度聚类DEScan.py

#@Software: PyCharm

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

# ============================数据探索================================== #
os.chdir('/Users/shan/Documents/Python学习/城市公交站点设置的优化分析/数据')
sns.set()
# ********************************************************************* #
# -------   DBScan密度聚类    -------- #
df_data = pd.read_csv("gjc.csv", sep=',', encoding='gbk')

# 聚类，半径为0.0011（度），3代表聚类中点的区域必须至少有3个才能聚成一类
db = DBSCAN(eps=0.0011, min_samples=3).fit(df_data.iloc[:, :2])
df_flag = pd.Series(db.labels_, name=('cluster'))

# axis中0是横向合并，1是纵向合并，若属性不对应就不会合并
df_cluster_result = pd.concat([df_data, df_flag], axis=1)
df_cluster_result.describe()

# 去掉噪声点
df_cluster_result = df_cluster_result[df_cluster_result["cluster"] >= 0]

# 站点聚类后散点图
df_station = df_cluster_result.drop_duplicates("cluster")  # 去重
df_station = df_station.reset_index()     # 增加了一列序列
plt.figure(figsize=(8, 4))
plt.scatter(df_station["经度"], df_station["纬度"], c=df_station["cluster"])
plt.show()

plt.scatter(df_cluster_result["经度"],
            df_cluster_result["纬度"],
            c=df_cluster_result["cluster"])
plt.show()