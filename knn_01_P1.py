from numpy import *
import operator

def CreateDataSet():
    """
        创建数据集与标签
    :return:
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
