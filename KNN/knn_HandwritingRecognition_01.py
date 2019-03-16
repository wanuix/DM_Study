"""
    将图像转换为测试向量
    使用分类起可以处理数字图像信息，将32x32的二进制图像矩阵转换为1x1024的向量
"""


from numpy import *

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


# 测试
test = img2vector('digits/testDigits/0_0.txt')
print(test[0, 33 : 64])