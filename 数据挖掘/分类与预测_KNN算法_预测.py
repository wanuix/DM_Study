"""
    KNN算法

    思想：
        一个样本在特征空间中，总会有k个最相似（即特征空间中最近邻）的样本
        大多数样本属于某一类，则该样本也属于这个类别

    算法流程：
        1、计算已知类别数据集中的点与当前点之间的距离
        2、按照距离递增次序排序
        3、选取当前点距离最小的k个点
        4、确定前k个点所在类别对应的出现频率
        5、返回前k个点出现频率最高的类别作为当前点点预测类别

    更多内容参考：
        https://blog.csdn.net/u014688145/article/details/64442996
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris


iris = load_iris()     # 加载数据
X = iris.data[:, :2]    # 为方便画图，仅采用数据的其中两个特征
y = iris.target
print(iris.DESCR)
print(iris.feature_names)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


# 初始化分类器对象
# 构建kNN分类器,第一个参数表示近邻数为3，算法为权重均匀的算法
clf = KNeighborsClassifier(n_neighbors=15, weights='uniform')
clf.fit(X, y)


# 画出决策边界，用不同颜色表示
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))


Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
print('测试样本错误率: %.2f%%' % ((1.0-clf.score(X, y))*100))

plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)    # 绘制预测结果图


plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)    # 补充训练数据点
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = 15, weights = 'uniform')")
plt.show()