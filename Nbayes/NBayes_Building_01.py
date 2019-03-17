"""
    朴素贝叶斯

    选择数据处于某一类概率更高的一类，作为其分类结果
    if P(c1|(x,y)) > P(c2|(x,y)):
        点(x,y)处的数据属于类别c1
    else:
        点(x,y)处的数据属于类别c2

    优点： 在数据较少的情况下仍然有效，可以处理多类别问题
    缺点： 对于输入数据的准备方式较为敏感
    适用数据类型： 标称型数据

    使用朴素贝叶斯进行文档分类
    我们使用RSS源收集数据
"""
"""
    准备数据：从文本中构建词向量
    词表到向量的转换函数
"""
from numpy import zeros

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

    return list(vocabSet)


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


# 测试


# listOposts, listClasses = loadDataSet()
# myVocabList = createVocabList(listOposts)
# print(myVocabList)
#
# # 测试myBocabList内的数据在listOposts的第一组数据中是否出现
# print(setOfWords2Vec(myVocabList, listOposts[0]))
#
# # 测试myBocabList内的数据在listOposts的第四组数据中是否出现
# print(setOfWords2Vec(myVocabList, listOposts[3]))