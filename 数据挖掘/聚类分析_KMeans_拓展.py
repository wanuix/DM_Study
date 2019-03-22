"""
    使用聚类实现对手数字对识别

    数据采用scikit-learn模块对digits数据集
"""
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics


dataSet = load_digits()
dataData = preprocessing.normalize(dataSet.data)
dataLables = dataSet.target


# DataTrain, DataTest, LabelsTrain, LabelsTest = \
#     train_test_split(dataData, dataLables, test_size=0.5, random_state=520)
kr = 0.6
DataTrain = dataData[:int(len(dataData)*kr)]
LabelsTrain = dataLables[:int(len(dataData)*kr)]
DataTest = dataData[int(len(dataData)*kr):-1]
LabelsTest = dataLables[int(len(dataData)*kr):-1]


Digits = KMeans(n_clusters=len(dataSet.target_names),
                     random_state=0)

Digits.fit(DataTrain, LabelsTrain)
Digits_Pre = Digits.predict(DataTest)
score = metrics.adjusted_rand_score(LabelsTest, Digits_Pre)

print("错误率为： %.2f%%" % ((1-score)*100))
# 最好结果——错误率为： 34.18%