"""
    朴素贝叶斯算法

    基于假设：特征之间是相互独立的

    贝叶斯分类在处理文档分类和垃圾邮件过滤有较好的表现

    高丝贝叶斯————处理连续数据

    多项式贝叶斯————处理多分类问题
"""
"""
    使用高丝贝叶斯模型
"""
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
clf = GaussianNB()
clf.fit(iris.data, iris.target)
y_pred = clf.predict(iris.data)
print("=" * 100)
print("总计点个数 %d， 贴错标签的点点数量 %d" % (iris.data.shape[0], (iris.target != y_pred).sum()))
print("-" * 100)
print("错误率： %.2f%%" % (((iris.target != y_pred).sum())/iris.data.shape[0]*100))
print("=" * 100)