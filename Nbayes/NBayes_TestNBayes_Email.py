"""
    示例：使用朴素贝叶斯过滤垃圾邮件
"""
from numpy import *
import NBayes_Building_03 as NB
from os import listdir

def textParse(bigString):  # input is big string, #output is word list
    """
        字符串切割函数
    :param bigString: 接收的大字符串
    :return:
    """
    import re

    # 利用正则表示式来切分句子
    listOfTokens = re.split(r'\W*', bigString)

    # 去掉少于两个字符的字符串，并将所有字符串转换为小写
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []

    # 导入并解析文本文件
    for i in range(1, 26):
        fr1 = open('email/spam/%d.txt' % i, 'r', encoding='UTF-8', errors='ignore') # 源码这里有错
        wordList1 = textParse(fr1.read())
        docList.append(wordList1)
        fullText.extend(wordList1)
        classList.append(1)

        fr2 = open('email/ham/%d.txt' % i, 'r', encoding='UTF-8', errors='ignore')
        wordList2 = textParse(fr2.read())
        docList.append(wordList2)
        fullText.extend(wordList2)
        classList.append(0)

    vocabList = NB.createVocabList(docList)  # create vocabulary

    trainingSet = list(range(50))   # 源码这里有错
    testSet = []  # create test set

    # 随机构建训练集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    trainMat = []
    trainClasses = []

    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(NB.bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = NB.trainNB0(array(trainMat), array(trainClasses))

    errorCount = 0

    # 对测试集分类
    for docIndex in testSet:  # classify the remaining items
        wordVector = NB.bagOfWords2VecMN(vocabList, docList[docIndex])
        if NB.classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error %s " % docList[docIndex])
    print(len(testSet))
    print('the number of errors is: %d the error rate is: %f%%' % (errorCount, float(errorCount) / len(testSet) * 100))

    # return vocabList,fullText

# 测试
spamTest()