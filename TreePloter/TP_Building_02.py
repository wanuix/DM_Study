"""
    对每个特征划分数据集的结果计算一次信息熵
    然后判断按照哪个特征划分数据集是最好的划分方式
"""
import TP_Building_01 as TP1


def splitDataSet(dataSet, axis, value):
    """
        按照给定特征对数据集进行划分
    :param dataSet:     数据集
    :param axis:    划分数据集的特征
    :param value:   特征值
    :return:
    """

    # 创建新的list对象
    retDataSet = []

    # 抽取数据集中的数据
    for featVec in dataSet:
        # 如果发现数据集中划分数据集的特征与特征值一致
        if featVec[axis] == value:
            # 將数据集到axis之前的列附到resucdFeatVec里
            reducedFeatVec = featVec[:axis]

            # 将axis以后的列附到reducedFeatVec里
            reducedFeatVec.extend(featVec[axis + 1:])

            # 將去除了指定特征列的数据集放在retDataSet里
            retDataSet.append(reducedFeatVec)

    # 返回划分后数据集（除去特征数据的数据集）
    return retDataSet


# 测试
# myDat, labels = TP1.createDataSet()
# print(splitDataSet(myDat, 0, 1))
# print("-" * 100)
# print(splitDataSet(myDat, 0, 0))
