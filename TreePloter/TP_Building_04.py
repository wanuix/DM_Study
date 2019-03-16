"""
    如果数据集已经处理了所有属性
    但是类标签已然不是唯一的
    此时我们需要决定如何定义该叶子节点
    在这种情况下
    我们通常会采用多数表决的方法决定该叶子节点的分类
"""

import operator
import TP_Building_03 as TP3


def majorityCnt(classList):
    """
        多数表决分类函数
    :param classList:  叶子分类列表
    :return:
    """

    # 建立一个数据字典，里面存储所有的类别
    classCount = {}

    # 遍历叶子分类列表所有值
    for vote in classList:
        # 如果有新的类别，则创立一个新的元素代表该种类
        if vote not in classCount.keys():
            classCount[vote] = 0

        # 否则该元素数量加一
        classCount[vote] += 1

    # 对数据集进行排序，第二行作为排序依据，从高到低排
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),
                              reverse=True)

    # 把第一个元素返回，即返回出现次数最多的那个类
    return sortedClassCount[0][0]

# 测试
myDat, labels = TP3.createDataSet()
bestFeature = TP3.chooseBestFeatureToSplit(myDat)
print(bestFeature)
classList = TP3.splitDataSet(myDat, 0, bestFeature)
classDic = [example[-1] for example in classList]
sortedClassCount = majorityCnt(classDic)
print(classList)
print(sortedClassCount)