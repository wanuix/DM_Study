"""
    使用k近邻算法识别手写数字
"""

from numpy import *
import numpy
from os import listdir
import knn_DatingWebsiteMatching_06 as knn


def img2vector(filename):
    """
        將图像转化为向量
    :param filename:  图片文件名
    :return:
    """

    # 打开文件
    fr = open(filename)

    # 构建预存一维向量，大小之所以为1024是因为图片大小为32*32=1*1024
    returnVector = zeros((1, 1024))

    # 將每行图像均转化为一维向量
    for i in range(32):
        # 按行读入每行数据
        lineStr = fr.readline()

        for j in range(32):
            # 將每行的每个数据依次存到一维向量中
            returnVector[0, 32 * i + j] = int(lineStr[j])

    # 返回处理好的一维向量
    return returnVector


def handwritingClassTest():
    """
        手写数字识别的knn实现
    :return:
    """

# 获取目录内容
    # 测试集的标签矩阵
    hwLabels = []

    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir('digits/trainingDigits')

    # 返回文件夹下文件的个数
    m = len(trainingFileList)

    # 初始化训练的Mat矩阵,测试集向量大小为训练数据个数*1024，即多少张图像，就有多少行，一行存一个图像
    trainingMat = numpy.zeros((m, 1024))

# 使用训练集，从文件名解析分类数字
    # 从文件名中解析出训练集的类别标签
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]

        # 第一个字符串存储标签，故取分离后的第一个元素，即相当于获取了该图像类别标签
        classNumber = int(fileNameStr.split('_')[0])

        # 将获得的类别标签添加到hwLabels中
        hwLabels.append(classNumber)

        filename = 'digits/trainingDigits/' + fileNameStr

        # 将每一个文件的1x1024数据存储到trainingMat中
        trainingMat[i, :] = img2vector(filename)

# 使用测试集
    # 返回testDigits目录下的文件列表
    testFileList = listdir('digits/testDigits')

    # 错误检测计数，初始值为0
    errorCount = 0.0

    # 测试数据的数量
    mTest = len(testFileList)

    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        # 获得文件的名字
        fileNameStr = testFileList[i]

        # 获得分类的数字标签
        classNumber = int(fileNameStr.split('_')[0])

        # 获得测试集的1x1024向量,用于训练
        vectorUnderTest = img2vector('digits/testDigits/%s' % (fileNameStr))

        # 利用knn获得预测结果
        classifierResult = knn.classify0(vectorUnderTest, trainingMat, hwLabels, 3)

        print("分类返回结果为 %d \t， 真实结果为 %d " % (classifierResult, classNumber))

        # 如果预测结果与实际结果不符，则错误数加一
        if (classifierResult != classNumber):
            errorCount += 1.0

    # 获取错误率
    print("总共错了 %d 个数据 \n， 错误率为 %f%%" % (errorCount, errorCount / mTest * 100))


# 测试
handwritingClassTest()