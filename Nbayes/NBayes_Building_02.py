"""
    训练算法：从词向量计算概率

    计算每个类别中的文档数目
    对每篇驯良文档：
        对每个类别：
            如果词条出现在文档中————增加该词条的计数值
            增加所有词条的计数值
        对每个类别：
            对每个词条：
                将该词条的数目除以总词条数目得到的条件概率
        返回每个类别的条件概率
"""

from numpy import *
import NBayes_Building_01 as BB1
import operator


def trainNB0(trainMatrix,trainCategory):
    """
        朴素贝叶斯分类器训练函数
    :param trainMatrix: 文档矩阵
    :param trainCategory: 每篇文档类标签所构成的向量
    :return: 词条属于普通的概率列表，词条属于侮辱性的概率列表，属于侮辱性概率
    """

    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])

    # 计算文档属于侮辱性文档的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)

    # 初始化概率
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)      # change to ones()
    p0Denom = 0.0
    p1Denom = 0.0               # change to 2.0

    # 遍历训练集中所有文档。
    # 一旦某个词（侮辱性/正常）出现，出现个数+1
    # 且所有文档总词数+1
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # 概率乘积取自然对数，解决计算溢出
    # p1Vect = log(p1Num/p1Denom)          #change to log()
    # p0Vect = log(p0Num/p0Denom)          #change to log()
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom


    return p0Vect,p1Vect,pAbusive


# 测试
listOPosts, listClasses = BB1.loadDataSet()
myVocabList = BB1.createVocabList(listOPosts)
trainMat = []
sortedMyVocabList = sorted(myVocabList, key=operator.itemgetter(0), reverse = False)
for postinDoc in listOPosts:
    trainMat.append(BB1.setOfWords2Vec(sortedMyVocabList, postinDoc))
p0V, p1V, pAb = trainNB0(trainMat, listClasses)

print(myVocabList)
print(sortedMyVocabList)
print('=' * 100)
print(pAb)
print('-' * 100)
print(p0V)
print('-' * 100)
print(p1V)