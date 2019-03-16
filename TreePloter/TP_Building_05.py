"""
    创建树的函数代码
"""

from math import log
import operator


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



def createTree(dataSet, labels):
    """
        创建树的函数代码
        创建分支的为代码函数createBranch():
        if so return 类标签;
        else:
            寻中划分数据集的最好特征
            划分数据集
            创建分支节点
                for 每个划分的子集
                    调用函数createBranch()并增加返回结果到分支节点中
            return 分支节点
    :param dataSet:     数据集
    :param labels:  标签列表
    :return:
    """

    # 以数据集的最后一列作为新的一个列表
    classList = [example[-1] for example in dataSet]

    # 如果分类列表完全相同
    if classList.count(classList[0]) == len(classList):
        # 停止继续划分
        return classList[0]

    # 如果遍历完所有特征，仍不能划分为唯一门类
    if len(dataSet[0]) == 1:
        # 返回出现出现次数最多的那个类标签
        return majorityCnt(classList)

    # 寻找选择最优特征
    bestFeat = chooseBestFeatureToSplit(dataSet)

    # 同时將最优特征的标签赋予bestFeatureLabel
    bestFeatLabel = labels[bestFeat]

    # 根据最优标签生成树
    myTree = {bestFeatLabel: {}}

    # 將刚刚生成树所使用的标签去掉
    del (labels[bestFeat])

    # 获取所有训练集中最优特征属性值
    featValues = [example[bestFeat] for example in dataSet]

    # 把重复的属性去掉，并放到uniqueVals里
    uniqueVals = set(featValues)

    # 遍历特征遍历的所有属性值
    for value in uniqueVals:
        # 先把原始标签数据完全复制，防止对原列表干扰
        subLabels = labels[:]

        # 调用递归函数继续划分
        myTree[bestFeatLabel][value] = \
            createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    # 返回决策树
    return myTree


# 测试
myDat, labels = createDataSet()
print(createTree(myDat, labels))