"""
    从文本文件中解析数据
"""

from numpy import *
import operator

def file2matrix(filename):
    """
        將文本记录转化为转化为Numpy矩阵
    :param filename:    文件名
    :return:
    """

    # 打开文件，地址变量存到fr里
    fr = open(filename)

    # 因为文件过大，这里推荐按行读取，并存到arrOlines里
    arrayOLines = fr.readlines()

    # 读取数据矩阵的行数
    numberOfLines = len(arrayOLines)

    # 建立 (文本文件行数， 3列) 的矩阵，以后整理的文件存在这里面
    returnMat = zeros((numberOfLines, 3))

    # 建立一个单列矩阵，存储其类
    classLabelVector = []

    # 索引值先清0
    index = 0

    # 按行读取文本，并依次给其贴标签
    for line in arrayOLines:
        # 將文本每一行首尾的空格去掉，截取掉所有的回车字符
        line = line.strip()

        # 矩阵中，每遇到一个'\t',便依次將这一部分赋给一个元素，
        # 使用tab字符\t将上一步得到的整行数据分割成一个元素列表
        listFromLine = line.split('\t')

        # 將每一行的前三个元素依次赋予之前预留矩阵空间，
        # 选取前3个元素，将它们存储到特征矩阵中
        returnMat[index, :] = listFromLine[0:3]

        # classLabelVector.append(int(float(listFromLine[-1])))
        # 对于每行最后一列，按照其值的不同，来给单列矩阵赋值
        if (listFromLine[-1] == 'largeDoses'):
            classLabelVector.append(3)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        index += 1  # 每执行一次，便向下一行再循环
    return returnMat, classLabelVector  # 返回两个矩阵，一个是三个特征组成的特征矩阵，另一个为类矩阵

datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
print(datingDataMat[0:5])
print("-" * 100)
print(datingLabels[0:5])