"""
    应用Python进行关联分析，包括对频繁数据集对探索、关联规则对建立和结果分析

    对于数据集Income，使用Apriori算法建立关联规则

    实现步骤

        1、获得数据集Income，查看数据集Income的前五个事项
        2、查看Income中各个项的支持度，
            并单独查看"age=14-34"的支持度
                 查看"sex=male"的支持度
                 查看支持度最大的前10项
        3、比较三个关联规则的数目
            1）以最小支持度为0.1，最小置信度为0.5建立Apriori关联规则，得到的关联规则记为rule1；
            2）以最小支持度为0.1，最小置信度为0.6建立Apriori关联规则，得到的关联规则记为rule2；
            3）以最小支持度为0.2，最小置信度为0.5建立Apriori关联规则，得到的关联规则记为rule3
"""
#-*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd

#
def connect_string(x, ms):
    """
        自定义连接函数，用于实现L_{k-1}到C_k的连接
    :param x: 符合支持度、置信度的对象
    :param ms: 连接符
    :return: 连接后的字符串
    """
    x = list(map(lambda i: sorted(i.split(ms)), x))
    l = len(x[0])
    r = []
    for i in range(len(x)):
        for j in range(i, len(x)):
            if x[i][:l - 1] == x[j][:l - 1] and x[i][l - 1] != x[j][l - 1]:
                r.append(x[i][:l - 1] + sorted([x[j][l - 1], x[i][l - 1]]))
    return r


#
def find_rule(d, support, confidence, ms=u'--'):
    """
        寻找关联规则的函数
    :param d: 源数据data （DataFrame）
    :param support: 最小支持度
    :param confidence:  最小置信度
    :param ms:  连接符，默认'--'，用来区分不同元素，如A--B。需要保证原始表格中不含有该字符
    :return:  关联度结果
    """
    # 定义输出结果
    result = pd.DataFrame(index=['support', 'confidence'])
    # 支持度序列
    support_series = 1.0 * d.sum() / len(d)
    # 初步根据支持度筛选
    column = list(support_series[support_series > support].index)
    k = 0

    while len(column) > 1:
        k = k + 1
        print(u'\n正在进行第%s次搜索...' % k)
        column = connect_string(column, ms)
        print(u'数目：%s...' % len(column))
        # 新一批支持度的计算函数
        sf = lambda i: d[i].prod(axis=1, numeric_only=True)

        # 创建连接数据，这一步耗时、耗内存最严重。当数据集较大时，可以考虑并行运算优化。
        d_2 = pd.DataFrame(list(map(sf, column)), index=[ms.join(i) for i in column]).T
        # print(d_2)

        # 计算连接后的支持度
        support_series_2 = 1.0 * d_2[[ms.join(i) for i in column]].sum() / len(d)
        # 新一轮支持度筛选
        column = list(support_series_2[support_series_2 > support].index)
        support_series = support_series.append(support_series_2)
        column2 = []

        # 遍历可能的推理，如{A,B,C}究竟是A+B-->C还是B+C-->A还是C+A-->B？
        for i in column:
            i = i.split(ms)
            for j in range(len(i)):
                column2.append(i[:j] + i[j + 1:] + i[j:j + 1])

        # 定义置信度序列
        cofidence_series = pd.Series(index=[ms.join(i) for i in column2])

        # 计算置信度序列
        for i in column2:
            cofidence_series[ms.join(i)] = support_series[ms.join(sorted(i))] / support_series[ms.join(i[:len(i) - 1])]

        # 置信度筛选
        for i in cofidence_series[cofidence_series > confidence].index:
            result[i] = 0.0
            result[i]['confidence'] = cofidence_series[i]
            result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]

    # 结果整理，输出
    result = result.T.sort_values(['confidence', 'support'], ascending=False)

    return result


# =====================================================================================
inputfile = 'Income.csv'
data = pd.read_csv(inputfile)
dataSet = data.iloc[:, 1:]
#
# 实现矩阵转换，空值用0填充
dataSet = pd.DataFrame(dataSet).fillna(0)
dataColumns = dataSet.columns
# 最小支持度
support = 0.1
# 最小置信度
confidence = 0.5
# 连接符，默认'--'，用来区分不同元素，如A--B。需要保证原始表格中不含有该字符
ms = '---'
# 保存结果
result = find_rule(dataSet, support, confidence, ms)
result = result.sort_values('support', ascending=False)
print(u'\n结果为：')
print(result)