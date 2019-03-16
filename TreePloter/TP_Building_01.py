"""
    在构造决策树时，需要解决第一个问题就是
    当前数据集上哪个特征在划分数据分类时起决定性作用
    所以我们首先必须评估每个特征

    创建分支的为代码函数createBranch():
        if so return 类标签;
        else:
            寻中划分数据集的最好特征
            划分数据集
            创建分支节点
                for 每个划分的子集
                    调用函数createBranch()并增加返回结果到分支节点中
            return 分支节点

    采用ID3算法划分数据集

    划分数据集的大原则是：将无序的数据变得更加有序
    在划分数据集之前之后信息发生的变化称为信息增益
    计算每个特征值划分数据集获得的信息增益，获得信息增益最高的特征就是最好的选择
"""
# =======================================================================================
"""
    计算给定数据集的香农熵
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

# 简单验证
# 建立简单的数据集
def createDataSet():
    # 这里不能写array，因为不是纯数
    dataSet = ([[1, 1, 'Yes'], [1, 1, 'Yes'], [1, 0, 'No'], [0, 1, 'No'], [0, 1, 'No']])
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# myDat, labels = createDataSet()
# shannonEnt= calcShannonEnt(myDat)
# print(shannonEnt)