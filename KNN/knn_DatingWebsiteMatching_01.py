s"""
    KNN算法的实现
"""
from numpy import *
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

#  测试

group, labels = createDataSet()
inx = [0, 0]
k = 3
print(classify0(inx, group, labels, k))