import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
# 合并数据集
#=====================================================================
#----------------------------------------------------------------------
# 数据库风格的DataFrame合并
# df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
#                     'data1': range(7)})
# df2 = pd.DataFrame({'key': ['a', 'b', 'd'],
#                     'data2': range(3)})
# print(pd.merge(df1, df2))  # 多对一合并
# print(pd.merge(df1, df2, on='key'))  # 以key列为指定列显示
# print(pd.merge(df1, df2, on='key', how='left'))  # 多对多合并
# print(pd.merge(df1, df2, on='key', how='inner'))  # 笛卡尔积
# print(pd.merge(df1, df2, on='key', how='outer'))  # 多个键的并集

#----------------------------------------------------------------------
# 轴向连接
# arr = np.arange(12).reshape((3, 4))
# print(arr)
# print(np.concatenate([arr, arr], axis=1))  # Numpy连接
# s1 = pd.Series([0, 1], index=['a', 'b'])
# s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
# s3 = pd.Series([5, 6, 7], index=['f', 'g', 'e'])
# print(pd.concat([s1, s2, s3]))  # Pandas连接
# df3 = pd.DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
# df4 = pd.DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
# print(pd.concat([df3, df4], ignore_index=True))
# "首届全国高校数据驱动创新研究大赛"优秀奖
# 基于豆瓣图书排行分析及预测用户行为和社会热点
# 利用python进行数据分析
#----------------------------------------------------------------------
# 合并重叠数据
# a = pd.Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
#               index=['f', 'e', 'd', 'c', 'b', 'a'])
# b = pd.Series(np.arange(len(a), dtype=np.float64),
#               index=['f', 'e', 'd', 'c', 'b', 'a'])
# b[-1] = np.nan
#
# print(np.where(pd.isnull(a), b, a))
# print("=" * 100)
# print(b[:-2].combine_first(a[2:]))


# 重塑和轴向旋转
#=====================================================================
#----------------------------------------------------------------------
# 重塑层次化索引
# data = pd.DataFrame(np.arange(6).reshape((2, 3)),
#                     index=pd.Index(['0hio', 'Colorado'], name='state'),
#                     columns=pd.Index(['one', 'two', 'three'], name='number'))
# print("=" * 100)
# print(data)
# print("=" * 100)
# print(data.T)   # 转置
# print("=" * 100)
# print(data.stack())  # 将列转为行
# print("=" * 100)
# print(data.unstack())  # 将行转为列
# print("=" * 100)
# # data.unstack().stack(dropna=False)  # 允许出现空值
# result = data.stack()
# df = pd.DataFrame({'left': result, 'right': result + 5},
#                   columns=pd.Index(['left', 'right'], name='side'))
# print(df.unstack('state'))
# print("=" * 100)
# print(df.unstack('state').stack('side'))
# print("=" * 100)

#----------------------------------------------------------------------
# 将"长格式"旋转为"宽格式"
# .pivot()


# 数据转换
#=====================================================================
#----------------------------------------------------------------------
# 移除重复数据
# data = pd.DataFrame({'k1': ['one']*3 + ['two']*4,
#                      'k2': [1, 1, 2, 3, 3, 4, 4]})
# print(data)
# print("1="*100)
# print(data.duplicated())  # 返回布尔值
# print("2="*100)
# print(data.drop_duplicates())  # 返回去重的DataFrame
# print("3="*100)
# data['v1'] = range(len(data))
# print(data)
# print("4="*100)
# print(data.drop_duplicates(['k1']))  # 根据k1列过滤重复项
# print("5="*100)
# print(data.drop_duplicates(['k1', 'k2'], keep='last'))  # 依据最后一个出现的值组合,这里不能用take_last=True

#----------------------------------------------------------------------
# 利用函数或映射进行数据转换
# data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami', 'corned beef', 'Bacon', 'pastrami', 'honey ham', 'nova lox'],
#                      'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
# print('1='*50)
# print(data)
# meat_to_animal = {'bacon': 'pig',
#                   'pulled pork': 'pig',
#                   'pastrami': 'cow',
#                   'corned beef': 'cow',
#                   'honey ham': 'pig',
#                   'nova lox': 'salmon'}
# # 添加一列表示该肉类食物来源的动物类型
# data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
# print("2="*50)
# print(data)
# # 传入一个能共完成全部这些工作的函数
# print("3="*50)
# print(data['food'].map(lambda x: meat_to_animal[x.lower()]))

#----------------------------------------------------------------------
# 替换值
# data = pd.Series([1., -999., 2., -999., -1000., 3.])
# print("1="*50)
# print(data)
# print("2="*50)
# print(data.replace([-999, -1000], np.nan))  # 利用replace产生一个用NAN替换后的Series
# # data.replace([-999, -1000], [np.nan, 0])  多替多
# # data.replace({-999: np.nan, -1000: 0})    字典多替多

#----------------------------------------------------------------------
# 重命名轴索引
# data = pd.DataFrame(np.arange(12).reshape((3, 4)),
#                     index=['Ohio', 'Colorado', 'New York'],
#                     columns=['one', 'two', 'three', 'four'])
# print("1="*50)
# print(data)
# print("2="*50)
# print(data.index.map(str.title))    # 首字母大写
# print(data.columns.map(str.upper))  # 全部大写
# print("3="*50)
# print(data.rename(index=str.title, columns=str.upper))
# print("4="*50)
# # rename可以结合字典对象实现对部分轴标签的更新
# print(data.rename(index={'Ohio': 'Indiana'},
#                   columns={'three': 'peekaboo'}))

#----------------------------------------------------------------------
# 离散化和面元划分
# ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
# bins = [18, 25, 34, 60, 100]
# cats = pd.cut(ages, bins)
# print("1="*50)
# print(cats)
# print(cats.codes)
# print(pd.value_counts(cats))
# print("2="*50)
# print(pd.cut(ages, bins, right=False))      # 右边是闭端（包括）
# print("3="*50)
# group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
# print(pd.cut(ages, bins, labels=group_names))
#
# #----------------------------------------------------------------------
# # 检测和过滤异常值
# np.random.seed(12345)
# data = pd.DataFrame(np.random.randn(1000, 4))
# print(data.describe())
# # 找出某列中绝对值大小超过3的值
# col = data[3]
# print(col[np.abs(col) > 3])
# # 选出含有"超过3或-3的值"的行，利用布尔行DataFrame以及any方法
# print(data[(np.abs(data) > 3).any(1)])
# # 将值限制在区间-3到3以内
# data[np.abs(data) > 3] = np.sign(data)*3
# print(data)
# print(data.describe())

#----------------------------------------------------------------------
# 排列和随机采样
# np.random.seed(12345)
# df = pd.DataFrame(np.arange(5*4).reshape(5, 4))
# sampler = np.random.permutation(5)
# print(df)
# print(sampler)
# print(df.take(sampler))
# # 非替换式采样
# print(df.take(np.random.permutation(len(df)))[:-1])

#----------------------------------------------------------------------
# 计算指标/哑变量
# df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
#                    'data1': range(6)})
# print(df)
# print(pd.get_dummies(df['key']))    # 转换为指标矩阵
# # 给指标DataFrame的列上加上一个前缀
# dummies = pd.get_dummies(df['key'], prefix='key')
# print(dummies)
# df_with_dummy = df[['data1']].join(dummies)
# print(df_with_dummy)


# 字符串操作
#=====================================================================
#----------------------------------------------------------------------
# 字符串对象方法
# val = 'a,b, guido'
# # 以逗号分割字符串，并修建空白符
# pieces = [x.strip() for x in val.split(',')]
# print('::'.join(pieces))
# print('::'.join(pieces).replace(':', '-'))

#----------------------------------------------------------------------
# 正则表达式


#=====================================================================
# 练习： 示例 USDA食品数据库
#----------------------------------------------------------------------
import json
db = json.load(open('database.json'))
print(len(db))
# db中每个条目都是一个含有某种食物全部数据的字典。
# nutrients字段是一个字典列表，
# 其中每个字典对应一种营养成分
# print(db[0].keys())
# print(db[0]['nutrients'][0])
nutrients = pd.DataFrame(db[0]['nutrients'])
# print(nutrients[:2])
# 取出食物的名称、分类、编号以及制造商等信息
info_keys = ['description', 'group', 'id', 'manufacturer']
info = pd.DataFrame(db, columns=info_keys)
# print(info[:2])
print("# 查看食物类别分布情况")
print(pd.value_counts(info.group))
print('='*100)
#----------------------------------------------------------------------
print("# 将所有食物的营养成分整合到一个大表中：")
print("# 将各食物的营养成分列表转换为一个DataFrame，")
print("# 并添加一个表示编号的列，")
print("# 然后将DataFrame添加到一个列表中")
print("# 最后通过concat将这些东西连接起来")
nutrients = []
for rec in db:
    fnuts = pd.DataFrame(rec['nutrients'])
    fnuts['id'] = rec['id']
    nutrients.append(fnuts)
nutrients = pd.concat(nutrients, ignore_index=True)
# print(nutrients)
# 丢弃DataFrame中的重复项
print('*'*50 + '丢弃DataFrame中的重复项' + '*'*50)
print(nutrients.duplicated().sum())
nutrients = nutrients.drop_duplicates()
# 重命名两个DataFrame
col_mapping_1 = {'description': 'food',
               'group': 'fgroup'}
info = info.rename(columns=col_mapping_1, copy=False)
col_mapping_2 = {'description': 'nutrient',
               'group': 'nutgroup'}
nutrients = nutrients.rename(columns=col_mapping_2, copy=False)
# 以某个属性为准，将两个数据集合并
ndata = pd.merge(nutrients, info, on='id', how='outer')
#----------------------------------------------------------------------
# 根据营养分类得出锌中位值
plt.figure(figsize=(8, 4))
result = ndata.groupby(['nutrient', 'fgroup'])['value'].quantile(0.5)
#result.head()
result['Zinc, Zn'].sort_values().plot(kind='barh')
plt.show()

# 营养成分最为丰富的食物
by_nutrients = ndata.groupby(['nutgroup', 'nutrient'])
get_maximum = lambda x: x.xs(x.value.idxmax())
get_minimum = lambda x: x.xs(x.value.idxmin())
max_foods = by_nutrients.apply(get_maximum)[['value', 'food']]
max_foods.food = max_foods.food.str[:50]
print('#'*100)
print(max_foods.loc['Amino Acids']['food'])

