"""
    选择最好的数据划分方式
"""

from math import log


def calcShannonEnt(dataSet):
    """
        计算香农熵
    :param dataSet: 数据集
    :return:
    """

    # 获取数据集的文件数目
    numEntries = len(dataSet)

    # 为所有可能分类创建字典
    labelCounts = {}

    # 依次在数据集当中將数据集的标签放到标签字典里
    for featVec in dataSet:
        # 依次取featVec里的最后一个元素，即标签元素
        currentLabels = featVec[-1]

        # 如果得到的标签不在已有的标签集内
        if currentLabels not in labelCounts.keys():
            # 创建一个新的标签
            labelCounts[currentLabels] = 0

            # 对对应的标签种类进行计数
        labelCounts[currentLabels] += 1

    # 初始化熵值
    shannonEnt = 0.0

    # 计算所有类别所有可能值包含的信息期望值
    for key in labelCounts:
        # 计算该分类的概率
        prob = float(labelCounts[key]) / numEntries

        # 计算期望
        shannonEnt -= prob * log(prob, 2)

    # 返回所有类别所有可能值包含的信息期望值
    return shannonEnt


# 建立简单的数据集
def createDataSet():
    # 这里不能写array，因为不是纯数
    dataSet = ([[1, 1, 'Yes'], [1, 1, 'Yes'], [1, 0, 'No'], [0, 1, 'No'], [0, 1, 'No']])
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


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


# =======================================================================================
def chooseBestFeatureToSplit(dataSet):
    """
        选择最好的数据分割点
    :param dataSet:     数据集
    :return:
    """

    # 对数据集第一行，即第一个例子的行数（特征个数），减1是为了方便数组计算
    lens = len(dataSet[0]) - 1

    # 算出原始香农熵
    baseShannonEnt = calcShannonEnt(dataSet)

    # 初始化信息增益
    bestGain = 0.0

    # 將最佳分割点值赋-1，不置0是因为0相当于第一个分割点，会引起误解
    bestFeature = -1

    # 循环各个特征找
    for i in range(lens):
        # 创集唯一的一个分类标签列表
        # 对于数据集里的所有特征遍历
        featList = [example[i] for example in dataSet]

        # 设置一个列表存放这些特征
        uniqueVals = set(featList)

        # 信息熵清零
        newEnt = 0.0

        # 遍历所有特征，根据特征不同来实现分割从而获取不同信息熵
        for value in uniqueVals:
            # 不同特征处分割
            subDataSet = splitDataSet(dataSet, i, value)

            # 计算该特征下的分类概率
            prob = len(subDataSet) / float(len(dataSet))

            # 获取此时的信息熵
            newEnt += prob * calcShannonEnt(subDataSet)

        # 获取信息增益（原始 - 新）
        infoGain = baseShannonEnt - newEnt

        # 如果信息增益比现有最好增益还大
        if (infoGain > bestGain):
            # 则取代他
            bestGain = infoGain

            # 并记下此时的分割位置
            bestFeature = i

    # 返回分割位置
    return bestFeature


# 测试
# myDat, labels = createDataSet()
# bestFeature = chooseBestFeatureToSplit(myDat)
# print(bestFeature)