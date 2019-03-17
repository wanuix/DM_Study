"""
    根据现实情况修改分类器
    建立朴素贝叶斯分类函数
"""
from numpy import *
import operator

# 创建实验样本
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    # 1代表侮辱性文字，0代表正常言论
    classVec = [0, 1, 0, 1, 0, 1]

    return postingList, classVec


def createVocabList(dataSet):
    """
        创建一个包含在所有文档中出现的不重复词的列表
    :param dataSet: 数据集
    :return:
    """

    vocabSet = set([])

    for document in dataSet:
        # 创建两个集合的并集
        vocabSet = vocabSet | set(document)

    # sortedMyVocabList = list(sorted(vocabSet, key=operator.itemgetter(0), reverse=False))
    # return  sortedMyVocabList
    return vocabSet

def setOfWords2Vec(vocabList, inputSet):
    """
        表示词汇表中的单词在输入文档中是否出现
    :param vocabList: 词汇表
    :param inputSet:  某个文档
    :return: 文档向量，向量每一个元素为1或0
    """

    # returnVec = [0] * len(vocabList)
    returnVec = zeros(len(vocabList))

    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)

    return returnVec


def bagOfWords2VecMN(vocabList, inputSet):
    """
        文档词袋模型，返回该词条在所有文档中出现的次数
    :param vocabList: 词汇表
    :param inputSet: 某个文档
    :return:
    """
    # returnVec = [0] * len(vocabList)
    returnVec = zeros(len(vocabList))
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


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
    p0Num = ones(numWords)
    p1Num = ones(numWords)      # change to ones()
    p0Denom = 2.0
    p1Denom = 2.0               # change to 2.0

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
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()

    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
        利用概率大小进行分类
    :param vec2Classify: 要进行分类的向量
    :param p0Vec: 处于正常对概率列表
    :param p1Vec: 处于侮辱对概率列表
    :param pClass1: 总类别侮辱性的概率
    :return: 判断依据，1是该类，0不是该类
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    """
        测试朴素贝叶斯算法
    :return:
    """

    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    print("=" * 100)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    print(pAb)
    print(p0V)
    print(p1V)
    print("=" * 100)


    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    print("-" * 100)

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


# 测试
# testingNB()