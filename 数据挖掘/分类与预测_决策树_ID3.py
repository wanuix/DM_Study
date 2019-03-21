"""
    使用ID3算法进行分类

    ID3算法实现：
       1、 对当前样本集合，计算所有属性的信息增益
       2、 选择信息增益最大的属性作为测试属性，把测试属性取值相同的样本划为同一个子样本集
       3、 若子样本集的类别属性只含有单个属性，则分支为子叶节点，
            判断其属性值并标上相应的符号之后返回调用处；
            否则对子样本集递归调用本算法
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC, export_graphviz
from graphviz import Digraph


# 读取数据
data = pd.read_csv('titanic_data.csv', encoding='utf-8')

# 舍弃ID列，不适合作为特征
data.drop(['PassengerId'], axis=1, inplace=True)

# 数据是类别标签，将其转换为数，用1表示男，0表示女。
data.loc[data['Sex'] == 'male', 'Sex'] = 1
data.loc[data['Sex'] == 'female', 'Sex'] = 0


# 缺失值处理
"""
缺失值的处理

    所有缺失值字段填充为 0：df.fillna(0)
    一定要十分注意的一点是，df.fillna() 操作默认（inplace=False）
    不是 inplace，也即不是对原始 data frame 直接操作修改的
    而是创建一个副本，对副本进行修改；
    
    df.fillna(0, inplace=True)
    df = df.fillna(0)
--------------------- 
作者：周小董 
来源：CSDN 
原文：https://blog.csdn.net/xc_zhou/article/details/84993074 
版权声明：本文为博主原创文章，转载请附上博文链接！
"""
data.fillna(int(data.Age.mean()), inplace=True)
"""
数值型数值(Numerical Data)

    方法一：fillna()函数
    
    df.fillna(0)：用0填充
    df.fillna(method=‘pad’)：用前一个数值填充
    df.fillna(df2.mean())：用该列均值填充

详见：https://blog.csdn.net/xc_zhou/article/details/84993074
"""


print(data.head(5))   # 查看数据
print("=" * 100)

X = data.iloc[:, 1:3]    # 为便于展示，未考虑年龄（最后一列）
y = data.iloc[:, 0]


dtc = DTC(criterion='entropy')    # 初始化决策树对象，基于信息熵
dtc.fit(X, y)    # 训练模型
print('输出准确率：%.2f%% ' % (dtc.score(X,y)*100))


# 可视化决策树，导出结果是一个dot文件，需要安装Graphviz才能转换为.pdf或.png格式
with open('tree.dot', 'w') as f:
    f = export_graphviz(dtc, feature_names=X.columns, out_file=f)

