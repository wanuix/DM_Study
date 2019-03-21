"""
    K-Means 算法

    算法用于分类

    在商业上常用于客户价值分析
    K-Means算法通过将样本划分为k个方差齐次的类来实现数据聚类
    该算法需要指定划分的类的个数

    算法实现：
        1、适当选取k个类的初始中心；
        2、在第k次的迭代中，对每一个样本x，求其到每个中心u的距离，将该样本归到距离最近的类中；
        3、对于每个类，通过均值计算出其中心u；
        4、如果通过2）3）的迭代更新每个中心u后，与更新前的值相差微小，则迭代终止，
           否则重复2）3）继续迭代

    参考资料:
        https://blog.csdn.net/Hisun_Gwen/article/details/72884606
"""
#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#指定默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.family'] = 'sans-serif'
#解决负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False


# 选取样本数量
n_samples = 1500
# 选取随机因子
random_state = 170
# 获取数据集
X, y = make_blobs(n_samples=n_samples, random_state=random_state)


# 聚类数量不正确时的效果
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)
plt.subplot(221)
plt.scatter(X[y_pred == 0][:, 0], X[y_pred == 0][:, 1], marker='x', color='b')
plt.scatter(X[y_pred == 1][:, 0], X[y_pred == 1][:, 1], marker='+', color='r')
plt.title("Incorrect Number of Blobs" + u"(聚类数量不正确时的效果)")


# 聚类数量正确时的效果
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)
plt.subplot(222)
plt.scatter(X[y_pred == 0][:, 0], X[y_pred == 0][:, 1], marker='x', color='b')
plt.scatter(X[y_pred == 1][:, 0], X[y_pred == 1][:, 1], marker='+', color='r')
plt.scatter(X[y_pred == 2][:, 0], X[y_pred == 2][:, 1], marker='1', color='m')
plt.title(u"Correct Number of Blobs(聚类数量正确时的效果)")


# 类间的方差存在差异的效果
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],  # 每个簇的标准差，衡量某簇数据点的分散程度
                                random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)
plt.subplot(223)
plt.scatter(X_varied[y_pred == 0][:, 0], X_varied[y_pred == 0][:, 1], marker='x', color='b')
plt.scatter(X_varied[y_pred == 1][:, 0], X_varied[y_pred == 1][:, 1], marker='+', color='r')
plt.scatter(X_varied[y_pred == 2][:, 0], X_varied[y_pred == 2][:, 1], marker='1', color='m')
plt.title(u"Unequal Variance(类间的方差存在差异的效果)")


# 类的规模差异较大的效果
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_filtered)
plt.subplot(224)
plt.scatter(X_filtered[y_pred == 0][:, 0], X_filtered[y_pred == 0][:, 1], marker='x', color='b')
plt.scatter(X_filtered[y_pred == 1][:, 0], X_filtered[y_pred == 1][:, 1], marker='+', color='r')
plt.scatter(X_filtered[y_pred == 2][:, 0], X_filtered[y_pred == 2][:, 1], marker='1', color='m')
plt.title(u"Unevenly Sized Blobs(类的规模差异较大的效果)")


plt.show()