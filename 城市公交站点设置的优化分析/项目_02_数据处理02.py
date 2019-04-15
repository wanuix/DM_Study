#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2019-04-15 15:32
#@Author  :Wanuix 
#@FileName: 项目_02_数据处理02.py

#@Software: PyCharm

import os
import pandas as pd

# ============================数据预处理================================== #
os.chdir('/Users/shan/Documents/Python学习/城市公交站点设置的优化分析/数据')
# ********************************************************************** #
# -------   循环读取文件（多文件操作）    -------- #
print("#"*50 + "提取68路多天GPS全部数据" + '#'*50)
file_list = os.listdir('./gps')
for filename in file_list:
    print("*"*50+"正在提取文件 %s 中的数据" % filename + "*"*50)
    data = pd.read_csv('./gps/' + filename, sep=',', encoding='gbk')
    print('=' * 100)
    if data.notnull().shape == data.shape:
        print("文件 %s 无空缺值" % filename)
        print('-' * 100)
        print("[属性]\t\t[是否有空值]")
        print(data.isnull().any())
    else:
        print("文件 %s 存在空缺值,个数为：%d个" %(filename,
              list(data.shape)[0] - list(data.notnull().shape)[0]))
    print('=' * 100)
    data_isduplicated = \
        data.duplicated(['业务时间', '卡片记录编码', '车牌号'], keep='first')
    temp = 0
    for i in range(data_isduplicated.shape[0]):
        if data_isduplicated[i]:
            temp = 1
            break
    del data_isduplicated
    if temp == 0:
        print("基于业务时间、卡片记录编码、车牌号属性查询，无重复数据，不需要去重")
    else:
        print("基于业务时间、卡片记录编码、车牌号属性查询，有重复数据，需要进行数据去重")
        data = data.drop_duplicates(
            ['业务时间', '卡片记录编码', '车牌号'], keep='first')
    del temp
    data_68 = data.loc[data.iloc[:, 4] == '68路', :]
    data_68.to_csv('data_68_gps_all.csv',
                   na_rep='NaN',
                   header=True,
                   index=False,
                   mode='a+')
    print("^" * 50 + "%s 中的数据已被提取且清洗完毕" % filename + "^" * 50)
print("#"*50 + "68路多天GPS全部数据提取完毕" + '#'*50)
