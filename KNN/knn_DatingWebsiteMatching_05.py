"""
    1 KNN算法的实现
    2 从文本文件中解析数据
    3 归一化数值
    4 测试分类器的效果
"""
from numpy import *
import numpy
import operator

def createDataSet():
    """
        建立简单点数据集
    :return:
    """
    group = array ([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
        该函数为简单kNN分类器
        对未知类别属性的数据集中的每个点依次执行以下操作：
            1、计算已知类别数据集中的点与当前点之间的距离；
            2、按照距离递增次序排序；
            3、选取与当前点距离最小的k个点；
            4、确定前k个点所在类别点出现频率；
            5、返回前k个点出现频率最高点类别作为当前点预测分类；
    :param inX:  用于分类的输入向量(测试向量)
    :param dataSet: 训练样本集
    :param labels:  标签向量
    :param k:   用于选择最近邻居的数目
    :return:
    """
# 首先计算已知类别数据集与当前点的距离
    # 读取数据集的行数，并把行数放到dataSetSize里，shape[]用来读取矩阵的行列数，shape[1]表示读取列数
    dataSetSize = dataSet.shape[0]

    """
    tile(inX,(dataSetSize,1))复制比较向量inX，tile的功能是告诉inX需要复制多少遍，这
    里复制成(dataSetSize行，一列)目的是把inX转化成与数据集相同大小，再与数据集矩阵相减，
    形成的差值矩阵存放在diffMat里
    """
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    # 注意这里是把矩阵李的各个元素依次平方，如（[-1,-1.1],[-1,-1]）执行该操作后为（[1,1.21],[1,1]）
    sqDiffMat = diffMat ** 2

    # 实现计算计算结果，axis表矩阵每一行元素相加(按行求和)，如（[1,1.21],[1,1]）,执行该操作后为（2.21，2）
    # 生成与dataset行相同点矩阵
    sqDistances = sqDiffMat.sum(axis=1)

    # 开根号
    distances = sqDistances ** 0.5

# 按照距离递增次序排序
    # 使用argsort排序，返回从小到大到“顺序值”   argsort()返回索引值
    # 如{2,4,1}返回{1,2,0}，依次为其顺序到索引
    sortedDisIndicies = distances.argsort()

    # 新建一个字典，用于计数
    classCount = {}

    # 选取与当前点距离最小的k个点
    # 按顺序对标签进行计数
    for i in range(k):
        # 按照之前排序值依次对标签进行计数
        # voteIlabel值是数据集labels中第sortedDisIndicies[i]个的标签
        voteIlabel = labels[sortedDisIndicies[i]]

        """
        对字典进行抓取，此时字典是空的,所以没有标签，现在將一个标签作为key，
        value就是label出现次数，因为数组从0开始，但计数从1开始，故需要加1
        """
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 返回一个列表按照第二个元素降序排列
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    # 返回出现次数最多到label值，即为当前点的预测分类
    return sortedClassCount[0][0]


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

    # 返回两个矩阵，一个是三个特征组成的特征矩阵，另一个为类矩阵
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
        对每个特征进行归一化处理
    :param dataSet: 数据集
    :return:
    """

    # 取数据集最小值
    minVals = dataSet.min(0)

    # 取数据集最大值
    maxVals = dataSet.max(0)

    # 函数计算可能的取值范围，取差值即为范围,并创建新的返回矩阵
    ranges = maxVals - minVals
    # 建立一个新0矩阵，其行数列数与数据集一致，处理后数据存这里
    normDataSet = zeros(numpy.shape(dataSet))

    # 读取数据集行数
    m = dataSet.shape[0]

    # 现有数据集减去最小值矩阵,
    # tile函数将变量内容复制成输入矩阵同样大小的矩阵
    normDataSet = dataSet - numpy.tile(minVals, (m, 1))

    # 归一化处理
    normDataSet = normDataSet / numpy.tile(ranges, (m, 1))

    return normDataSet, ranges, minVals


def datingClassTest():
    """
        分类器针对约会网站的测试代码
    :return:
    """

    # 载入数据
    filename = "datingTestSet.txt"

    # 提取10%的数据为测试数据集
    hoRatio = 0.10

    # 將文本数据转化为numpy矩阵
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')

    # 数据归一化处理
    normMat, ranges, minVals = autoNorm(datingDataMat)

    # 获取数据矩阵行数
    m = normMat.shape[0]

    # 获取测试集数据行数
    numTestVecs = int(m * hoRatio)

    # 初始化出错累计变量
    errorCount = 0.0

    # 获取测试集的错误率
    for i in range(numTestVecs):
        # 执行之前的分类器，并將分类结果放在classifierResult里
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 4)
        print('the classifier came back with:%d,the real answer is %d'
              % (classifierResult, datingLabels[i]))

        # 如果分类器的结果与标签值不一致，则將错误数加一
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0

    # 输出错误率
    print('the total error rate is %%' '%f' % (errorCount / float(numTestVecs) * 100))


#  测试
#---------------------------------------------------------------------------
# datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
# normMat, ranges, minVals = autoNorm(datingDataMat)
# print(normMat)
# print(ranges)
# print(minVals)
datingClassTest()

