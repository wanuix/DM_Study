"""
    DBSCAN聚类算法

    密度聚类算法

    基本思想：
        将类定义为密度相连的点点最大集合，通过在样本空间中不断寻找最大集合从而完成聚类

    该算法能在带噪声点样本空间中发现任意形状点聚类并排除噪声

    算法实现：
        1、定义半径ipsilon和MinPts（给定对象ipsilon邻域内点样本点数大于设定点MinPts，则称该对象为核心对象）
        2、从对象集合D中抽取未被访问过的点样本点q
        3、检验该样本点是否为核心对象，如果是则进入步骤4，否则返回步骤2
        4、找出该样本点所有从点密度可达点对象，构成聚类C。
            注意：构成点聚类C点边界对象都是非核心对象（否则继续进行深度搜索）
                以及在此过程中所有被访问过点对象都会被标记为已访问
        5、如果全部样本点都已被访问，则结束算法，否则返回步骤2

    DBSCAN算法能够过滤低密度区域，发现稠密样本点

    与K-Mean算法比，DBSCAN算法不需要指定划分点聚类个数，算法能够返回这个信息

    DBSCAN算法最大点优点是可以过滤噪声

    如果我们难以预知聚类数量，我们应该放弃K-Mean而选择DBSCAN
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
"""
如果不加一下两句
会出现以下警告信息：
/anaconda3/lib/python3.7/site-packages/sklearn/metrics/cluster/supervised.py:732: 
    FutureWarning: The behavior of AMI will change in version 0.22. 
    To match the behavior of 'v_measure_score', AMI will use average_method='arithmetic' by default.
      FutureWarning)
虽然这个警告信息并不影响正常结果的输出
但是看起来让人觉得很不舒服
于是就想着一个方法把这个警告信息给去了。
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=732)

##############################################################################
# 获取make_blobs数据
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)
# 数据预处理
X = StandardScaler().fit_transform(X)


##############################################################################
# 执行DBSCAN算法
db = DBSCAN(eps=0.3, min_samples=10).fit(X)  # 定义半径ipsilon和MinPts
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# 标记核心对象,后面作图需要用到
core_samples_mask[db.core_sample_indices_] = True
# 算法得出的聚类标签,-1代表样本点是噪声点,其余值表示样本点所属的类
labels = db.labels_
# 获取聚类数量
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


##############################################################################
# 这些度量均是越大越好
# 参考：
#   https://www.jianshu.com/p/841ecdaab847
#   https://blog.csdn.net/sinat_26917383/article/details/70577710
# --------------------------------------------------------------------------------
# 外部度量：

# 输出算法性能的信息
print('Estimated number of clusters: %d' % n_clusters_)

# 同质性：所有聚类都只包含属于单个类成员点数据点
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))

# 完整性：作为给定类点成员的所有数据点是相同集群的元素
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))

# 两者的调和平均V-measure，上面两者的一种折衷：
# v = 2 * (homogeneity * completeness) / (homogeneity + completeness)
# 可以作为聚类结果的一种度量。
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))

# --------------------------------------------------------------------------------
# 在真实的分群label不知道的情况下(内部度量)：

# CH指标通过计算类中各点与类中心的距离平方和来度量类内的紧密度
# 通过计算各类中心点与数据集中心点距离平方和来度量数据集的分离度
# CH指标由分离度与紧密度的比值得到
# 从而，CH越大代表着类自身越紧密，类与类之间越分散，即更优的聚类结果。
print("Calinski-Harabaz Index: %.3f" % metrics.calinski_harabaz_score(X, labels))

# 调整兰德系数。
# ARI取值范围为[-1,1]
# 从广义的角度来讲，ARI衡量的是两个数据分布的吻合程度
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))

# 调整互信息。
# 利用基于互信息的方法来衡量聚类效果需要实际类别信息
# MI与NMI取值范围为[0,1],AMI取值范围为[-1,1]。
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))

# 轮廓系数
# silhouette_sample
# 对于一个样本点(b - a)/max(a, b)
# a平均类内距离，b样本点到与其最近的非此类的距离。
# silihouette_score返回的是所有样本的该值,取值范围为[-1,1]。
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))


##############################################################################
# 绘图
import matplotlib.pyplot as plt

# 黑色用作标记噪声点
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

i = -1
# 标记样式,x点表示噪声点
marker = ['v', '^', 'o', 'x']
for k, col in zip(unique_labels, colors):
    if k == -1:
        # 黑色表示标记噪声点.
        col = 'k'

    class_member_mask = (labels == k)

    i += 1
    if (i >= len(unique_labels)):
        i = 0

    # 绘制核心对象
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], marker[i], markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    # 绘制非核心对象
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], marker[i], markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()