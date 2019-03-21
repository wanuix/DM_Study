"""
    系统聚类算法（层次聚类，系谱聚类）

    通过合并或分割类，生成嵌套的集群

    基本思想：
        先将样本看作各自一类，定义类间距离的计算方法
        选择距离最小的一对类合并成为一个新的类
        重新计算类间的距离
        再将距离最近的两类合并
        如此最终合成一类

    算法实现：
        1、初始化，定义样本间距离和类间距离的计算方法，将每个样本点各自设为一类
        2、计算任意两个类间的距离d，将最短距离的两个类合并
        3、如果已经聚为k类，则算法停止，否则重复2
"""


import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

plt.figure(figsize=(12, 12))

# 选取样本数量
n_samples = 1500
# 选取随机因子
random_state = 170
# 获取数据集
X, y = make_blobs(n_samples=n_samples, random_state=random_state)


"""
class sklearn.cluster.AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity=’euclidean’, verbose=False)

函数参数

    damping : float, optional, default: 0.5,阻尼系数,默认值0.5

    max_iter : int, optional, default: 200,最大迭代次数,默认值是200

    convergence_iter : int, optional, default: 15,在停止收敛的估计集群数量上没有变化的迭代次数。默认15

    copy : boolean, optional, default: True,布尔值,可选,默认为true,即允许对输入数据的复制

    preference : array-like, shape (n_samples,) or float, optional,近似数组,每个点的偏好 - 具有较大偏好值的点更可能被选为聚类的中心点。 簇的数量，即集群的数量受输入偏好值的影响。 如果该项未作为参数，则选择输入相似度的中位数作为偏好

    affinity : string, optional, default=``euclidean``目前支持计算预欧几里得距离。 即点之间的负平方欧氏距离。

    verbose : boolean, optional, default: False

成员属性:

    cluster_centers_indices_ : array, shape (n_clusters,)聚类的中心索引,Indices of cluster centers

    cluster_centers_ : array, shape (n_clusters, n_features)聚类中心（如果亲和力=预先计算）。

    labels_ : array, shape (n_samples,)每个点的标签

    affinity_matrix_ : array, shape (n_samples, n_samples)存储拟合中使用的亲和度矩阵。

    n_iter_ : int,收敛的迭代次数。

近邻传播算法的复杂度是点数的二次

成员方法:

fit(X[, y])    从负欧氏距离创建相似度矩阵，然后应用于近邻传播聚类。

fit_predict    在X上执行聚类并返回聚类标签

get_params    获取此估算器的参数

predict(X)    预测X中每个样本所属的最近聚类

set_params    设置此估算器的参数。

__init__(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity=’euclidean’, verbose=False)        初始化函数


--------------------- 
作者：manjhOK 
来源：CSDN 
原文：https://blog.csdn.net/manjhOK/article/details/79586791 
版权声明：本文为博主原创文章，转载请附上博文链接！
"""
# 聚类数量不正确时的效果
y_pred = AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters=2).fit_predict(X)
# 选取欧几里德距离和离差平均和法
plt.subplot(221)
plt.scatter(X[y_pred == 0][:, 0], X[y_pred == 0][:, 1], marker='x', color='b')
plt.scatter(X[y_pred == 1][:, 0], X[y_pred == 1][:, 1], marker='+', color='r')
plt.title("Incorrect Number of Blobs")


# 聚类数量正确时的效果
y_pred = AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters=3).fit_predict(X)
plt.subplot(222)
plt.scatter(X[y_pred == 0][:, 0], X[y_pred == 0][:, 1], marker='x', color='b')
plt.scatter(X[y_pred == 1][:, 0], X[y_pred == 1][:, 1], marker='+', color='r')
plt.scatter(X[y_pred == 2][:, 0], X[y_pred == 2][:, 1], marker='1', color='m')
plt.title("Correct Number of Blobs")


# 类间的方差存在差异的效果
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
y_pred = AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters=3).fit_predict(X_varied)
plt.subplot(223)
plt.scatter(X_varied[y_pred == 0][:, 0], X_varied[y_pred == 0][:, 1], marker='x', color='b')
plt.scatter(X_varied[y_pred == 1][:, 0], X_varied[y_pred == 1][:, 1], marker='+', color='r')
plt.scatter(X_varied[y_pred == 2][:, 0], X_varied[y_pred == 2][:, 1], marker='1', color='m')
plt.title("Unequal Variance")

# 类的规模差异较大的效果
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_pred = AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters=3).fit_predict(X_filtered)

plt.subplot(224)
plt.scatter(X_filtered[y_pred == 0][:, 0], X_filtered[y_pred == 0][:, 1], marker='x', color='b')
plt.scatter(X_filtered[y_pred == 1][:, 0], X_filtered[y_pred == 1][:, 1], marker='+', color='r')
plt.scatter(X_filtered[y_pred == 2][:, 0], X_filtered[y_pred == 2][:, 1], marker='1', color='m')
plt.title("Unevenly Sized Blobs")

plt.show()