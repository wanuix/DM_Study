#-*- coding: utf-8 -*-
#使用基于UBCF算法对电影进行推荐
from __future__ import print_function
import pandas as pd
# http://pandas.pydata.org/pandas-docs/stable/search.html?q=.copy%28%29&check_keywords=yes&area=default
import math
"""
    智能推荐系统
    
    可以为每个用户实现个性化的推荐结果，让每个用户更便捷的获取信息
    
    主要推荐技术：
        基于用户的协同过滤推荐
        基于物品的协同过滤推荐
    
    基于用户的协同过滤推荐基本思想：
        基于用户对物品的偏好找到邻居用户，然后将邻居用户喜欢的推荐给当前用户
    
    基于物品的协同过滤推荐基本思想：
        将一个用户对所有物品的偏好作为一个向量来计算用户之间的相似度
        找到K邻居后，根据邻居的相似度权重以及他们对物品的偏好
        预测当前用户没有偏好的未涉及物品
        计算得到一个排序物品列表作为推荐
    
    原理区别：
        基于物品的协同过滤原理和基于用户的协同过滤原理类似
        只是在计算邻居时采用物品本身，而不是从用户的角度
        即基于用户对物品的偏好找到相似的物品
        然后根据用户的历史偏好，推荐相似的物品给他
    
    基于用户的协同过滤算法实现：
        1、计算用户之间的相似度
            皮尔逊相关系数
            基于欧几里得距离的相似度
            余弦相似度
        2、计算用户u对未评分商品的预测分值
        3、基于对未评分商品的预测分值排序，得到推荐商品列表
"""


def prediction(df, userdf, Nn = 15):
    """
        获取所有用户对所没看过的电影对预测评分
    :param df:  训练集
    :param userdf:  用户集
    :param Nn: Nn邻居个数
    :return: 计算用户对未评分商品的预测分值
    """
    # 将df转置后，再计算列的成对相关性
    corr = df.T.corr()
    # 返回数组的副本
    rats = userdf.copy()

    # 遍历用户集行标签
    for usrid in userdf.index:
        # 训练集每行为NULL的数据集
        dfnull = df.loc[usrid][df.loc[usrid].isnull()]
        # traindf.iloc[0][traindf.loc[usrid].isnull()]
        # traindf.iloc[0][traindf.loc[usrid].notnull()]

        # 用户usrid对所评价过电影对评价评分对平均值（每行非空数据的平均值）
        usrv = df.loc[usrid].mean()

        # 遍历训练集每行为NULL的数据集
        for i in range(len(dfnull)):
            # 训练集第 dfnull.index[i] 列数据的非空数据集
            # 返回的是布尔集
            # TODO 感觉此处代码有问题，返回长度不对
            nft = (df[dfnull.index[i]]).notnull()
            # 感觉下面修改后的代码是对的
            # nft = df.T.loc[dfnull.index[i]][df.T.loc[dfnull.index[i]].notnull()]

            # 获取usrid这个用户对看过dfnull.index[i]电影的邻居列表
            if(Nn <= len(nft)):
                # traindf.T.loc[dfnull.index[0]][traindf.T.loc[dfnull.index[0]].notnull()]
                # 训练集第dfnull.index[i] 列数据的非空数据集的前Nn个数据
                # 返回的是对这个dfnull.index[i]电影评过分对user对应的电影评分rat数据集
                nlist = df[dfnull.index[i]][nft][:Nn]
            else:
                nlist = df[dfnull.index[i]][nft][:len(nft)]

            # 训练集相关度矩阵第usrid行中列值为nlist.index的数据集中
            # 非空项对应的nlist数据集中的内容
            # 剔除与usrid不想关的用户对于该电影的评分
            nlist = nlist[corr.loc[usrid, nlist.index].notnull()]

            nratsum = 0
            corsum = 0

            # 如果nlist不为空
            if(0 != nlist.size):
                # 邻居所评价过的电影的评分的平均值数据集
                nv = df.loc[nlist.index, :].T.mean()

                for index in nlist.index:
                    # usrid与邻居的相关系数
                    ncor = corr.loc[usrid, index]

                    # 邻居nlist.index对usrid没看过的电影dfnull.index[i]的评分
                    # 减
                    # 该邻居对所有电影评分平均值的值
                    # 乘
                    # usrid与邻居nlist.index的相关系数
                    # 再累加nrastsum
                    # 预测用户usrid对电影i的评分的计算公式的分子
                    nratsum += ncor * (df[dfnull.index[i]][index]-nv[index])

                    # usrid与邻居的相关系数的累加
                    # 预测用户usrid对电影i的评分的计算公式的分母（此时还没有开根）
                    corsum += abs(ncor)

                if(corsum != 0):
                    # 访问行/列标签对的单个值。
                    # 用户usrid对电影i对评分
                    rats.at[usrid, dfnull.index[i]] = usrv + nratsum/(math.sqrt(corsum))
                else:
                    rats.at[usrid, dfnull.index[i]] = usrv
            else:
                rats.at[usrid, dfnull.index[i]] = None
    return rats


def recomm(df, userdf, Nn = 15, TopN = 3):
    """
        获取预测评分和推荐列表
    :param df: 训练集
    :param userdf: 用户集
    :param Nn:
    :param TopN:
    :return: 返回预测评分矩阵ratings 与 对每个用户对推荐结果recomm
    """

    # 获取预测评分
    ratings = prediction(df, userdf, Nn)


    # 存放推荐结果
    recomm = []
    for usrid in userdf.index:
        # 获取按NULL值获取未评分项
        ratft = userdf.loc[usrid].isnull()

        # 获得用户usrid对未评分项对预测评分
        ratnull = ratings.loc[usrid][ratft]

        # 对预测评分进行排序
        # 如果非评分项超过3个
        if len(ratnull) >= TopN:
            # .sort_values 按任一轴的值排序
            # 默认为升序
            # ascending=False 为降序排序
            sortlist = (ratnull.sort_values(ascending=False)).index[:TopN]
        else:
            sortlist=ratnull.sort_values(ascending=False).index[:len(ratnull)]

        recomm.append(sortlist)

    return ratings, recomm


print("\n--------------使用基于UBCF算法对电影进行推荐 运行中... -----------\n")

# 读取数据
traindata = pd.read_csv('u1.base', sep='\t', header=None, index_col=None)
testdata = pd.read_csv('u1.test', sep='\t', header=None, index_col=None)


# 删除时间标签列(删除最后一列)
traindata.drop(3, axis=1, inplace=True)
testdata.drop(3, axis=1, inplace=True)


# 行与列重新命名
traindata.rename(columns={0: 'userid', 1: 'movid', 2: 'rat'}, inplace=True)
testdata.rename(columns={0: 'userid', 1: 'movid', 2: 'rat'}, inplace=True)


# 以userid作为行x，以movid作为列y，以rat作为值，构建矩阵，空值为对应位置(x,y)无rat值
# http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot.html
traindf = traindata.pivot(index='userid', columns='movid', values='rat')
testdf = testdata.pivot(index='userid', columns='movid', values='rat')


# 以 'user + traindf.index(行号)' 重新命名行名
# 以 'mov + traindf.columns(列号)' 重新命名列名
traindf.rename(index={i: 'usr%d' %(i) for i in traindf.index}, inplace=True)
traindf.rename(columns={i: 'mov%d' %(i) for i in traindf.columns}, inplace=True)
testdf.rename(index={i: 'usr%d' %(i) for i in testdf.index}, inplace=True)
testdf.rename(columns={i: 'mov%d' %(i) for i in testdf.columns}, inplace=True)


# .loc 按标签或布尔数组访问一组行和列
# 通过 test测试集的行标签 索引 train训练集的行数据
# 测试集的用户在训练集中也存在的数据集
userdf = traindf.loc[testdf.index]


# 获取预测评分和推荐列表
trainrats, trainrecomm = recomm(traindf, userdf)

print(trainrecomm)
