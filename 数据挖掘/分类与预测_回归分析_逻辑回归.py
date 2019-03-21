"""
    罗辑回归sklearn实现
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import  train_test_split


# 导入csv数据
data = pd.read_csv('LogisticRegression.csv', encoding='utf-8')


# 将数据型变量进行独热编码
"""
    离散特征编码（one-hot 独热编码）
    离散特征的编码分为两种情况：
    
    1、离散特征的取值之间没有大小的意义，比如color：[red,blue],那么就使用one-hot编码
    
    2、离散特征的取值有大小的意义，比如size:[X,XL,XXL],那么就使用数值的映射{X:1,XL:2,XXL:3}
    
    使用pandas可以很方便的对离散型特征进行one-hot编码
"""
"""
get_dummies方法：
    
    data：array-like，Series或DataFrame
    prefix：string，字符串列表或字符串dict，默认为None
            附加DataFrame列名称的字符串在DataFrame上调用get_dummies时，传递一个长度等于列数的列表。
            或者，prefix可以是将列名称映射到前缀的字典。
    prefix_sep：string，默认为'_'
            如果附加前缀，分隔符/分隔符要使用。或者传递与前缀一样的列表或字典。
    dummy_na：bool，默认为False
            如果忽略False NaN，则添加一列以指示NaN。
    columns：类似列表，默认为无
            要编码的DataFrame中的列名称。如果列为None，则将转换具有object或category dtype的所有列。
    sparse：bool，默认为False
            虚拟列是否应该稀疏。如果数据是Series或者包含所有列，则返回SparseDataFrame。
            否则返回带有一些SparseBlocks的DataFrame。
    drop_first：bool，默认为False
            是否通过删除第一级别从k分类级别获得k-1个假人。
            
    New in version 0.18.0.
    Returns
    ——-
    dummies : DataFrame or SparseDataFrame
"""
data_dum = pd.get_dummies(data, prefix='rank', columns=['rank'], drop_first=True)


print("=" * 100)
print(data.head(5))
print("-" * 100)
print(data_dum.tail(5)) # 看后5行内容
print("=" * 100)


# 切分数据集与测试集
"""
    train_test_split 是交叉验证中常见的函数
    功能是从样品中随机的按比例选取 train data 和 test data
    一般格式：
        x_train, x_test, y_train, y_test = 
            cross_validation.train_test_split
                (
                    train_data, 
                    train_target,
                    test_size = 0.5,
                    random_state = 0
                )
        train_data : 所要划分的样本特征集
        train_target : 所要划分的样本结果
        test_size : 样本占比，如果是整数为样本的数量
        random_state: 随机数的种子（ =1 每组随机数相同； =0 每组随机数不同）
        随机数种子：其实就是该组随机数的编号
                在需要重复试验的时候，保证得到一组一样的随机数
                比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的
                但填0或不填，每次都会不一样。随机数的产生取决于种子
                随机数和种子之间的关系遵从以下两个规则：
                    种子不同，产生不同的随机数；
                    种子相同，即使实例不同也产生相同的随机数。
"""
X_train, X_test, y_train, y_test = \
    train_test_split(data_dum.iloc[:, 1:],
                     data_dum.iloc[:, 0],
                     test_size=.1,
                     random_state=520)
"""
    loc：通过行标签索引行数据
    iloc: 通过行号索引行数据
    ix:通过行号或者行标签来索引行数据
    但是！！！
    ix被弃用，可按照需求采用.loc和.iloc索引数据
"""


# Logistic逻辑回归模型
# lr = LogisticRegression()
lr = LogisticRegression(solver='liblinear')
"""
这里遇到警告问题，下面是百度结果

在Python中利用Logistic回归算法进行数据建模，本来算是比较常见的事情
但结果“阴沟里翻船”，一上来就遇到了报警提示。

在PyCharm中，使用python的sklearn.linear_model.LogisticRegression进行实例化时
model=LogisticRegression()，就提示了以下警告信息：

FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning。

虽然警告信息并不影响代码运行，但输出窗口异常明显的几行红字提醒，我总觉得代码的心里也不会很爽快。

 

问题分析：

FutureWarning是语言或者库中将来可能改变的有关警告。

根据报警信息和参考相关文档，“Default will change from 'liblinear' to 'lbfgs' in 0.22.”，默认的solver参数在0.22版本中，将会由“liblinear”变为“lbfgs”，且指定solver参数可以消除该warning。

这是代码在发出警告，将来代码运行时如果没有及时关注到版本的问题，可能solver的参数会发生改变。所以，最安全的方法并不是通过ignore消除警告，而是指定一个solver参数。

参阅官方文档：

solver : str, {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, \default: 'liblinear'.

        Algorithm to use in the optimization problem.

        - For small datasets, 'liblinear' is a good choice, whereas 'sag' and

          'saga' are faster for large ones.

        - For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'

          handle multinomial loss; 'liblinear' is limited to one-versus-rest

          schemes.

        - 'newton-cg', 'lbfgs' and 'sag' only handle L2 penalty, whereas

          'liblinear' and 'saga' handle L1 penalty.

 

LogisticRegerssion算法的solver仅支持以下几个参数'liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'。

 

解决方法：

传入参数后即可消除警告：model=LogisticRegression(solver=’liblinear’)  。


扩展

其他消除警告的方法：

import warnings

warnings.filterwarnings("ignore")
--------------------- 
作者：linzhjbtx 
来源：CSDN 
原文：https://blog.csdn.net/linzhjbtx/article/details/85331200 
版权声明：本文为博主原创文章，转载请附上博文链接！
"""


# 训练模型
lr.fit(X_train, y_train)


print('逻辑回归的准确率为：{0:.2f}%'.format(lr.score(X_test, y_test) * 100))