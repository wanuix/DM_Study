import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from numpy.random import randn
from mpl_toolkits.basemap import Basemap


# Page 8-246 点线图
# df = pd.DataFrame(randn(20, 4).cumsum(0),
#                   columns=['A', 'B', 'C', 'D'],
#                   index=np.arange(0, 100, 5))
sns.set()
# df.plot()
# plt.show()


# Page 8-248 横竖柱状图
# fig, axes = plt.subplots(2, 1)
# data = pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))
# data.plot(kind='bar', ax=axes[0], color='g', alpha=0.7)
# data.plot(kind='barh', ax=axes[1], color='b', alpha=0.7)
# plt.show()


# Page 8-249 分组柱状图
# df = pd.DataFrame(np.random.rand(6, 4),
#                   index=['one', 'two', 'tree', 'four', 'five', 'six'],
#                   columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
# print(df)
# df.plot(kind='bar')
# plt.show()


# Page 8-250 堆积柱状图
# df = pd.DataFrame(np.random.rand(6, 4),
#                    index=['one', 'two', 'tree', 'four', 'five', 'six'],
#                    columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
# print(df)
# df.plot(kind='barh', stacked=True, alpha=0.5)
# plt.show()


# Page 8-253 带有密度估计的规格化直方图
# comp1 = np.random.normal(0, 1, size=200)  # 正态分布
# comp2 = np.random.normal(20, 2, size=200)
# values = pd.Series(np.concatenate([comp1, comp2]))
# values.hist(bins=100, alpha=0.3, color='g', normed=True)
# values.plot(kind='kde', style='b--')
# plt.show()
# print(comp1)
# print('-' * 100)
# print(comp2)


# Page 8-254 StatsModels Macro Data 的散布图矩阵
# macro = pd.read_csv('macrodata.csv')
# data = macro[['cpi', 'm1', 'tbilrate', 'unemp']]
# trans_data = np.log(data).diff().dropna()
# print(trans_data[-5:])
# plt.scatter(trans_data['m1'], trans_data['unemp'])
# plt.title('Change in log %s vs. log %s' % ('m1', 'unemp'))
# # plt.show()
# pd.scatter_matrix(trans_data, diagonal='kde', color='g', alpha=0.3)
# plt.show()

#============================================================================
# 训练：绘制地图 图形化显示海底地震危机数据
# Page 8-259
data = pd.read_csv('Haiti.csv')
print(data.describe())
print('=' * 100)
data = data[(data.LATITUDE > 18) &
            (data.LATITUDE < 20) &
            (data.LONGITUDE > -75) &
            (data.LONGITUDE < -70) &
            data.CATEGORY.notnull()]
print(data.describe())
print('=' * 100)
def to_cat_list(catstr):
    """
        按照"，"号切割字符串
    :param catstr: 需要被切割的字符串
    :return:
    """
    stripped = (x.strip() for x in catstr.split(','))
    return [x for x in stripped if x]
def get_all_caegories(cat_series):
    """
        获得所有分类的列表
    :param cat_series: 列表
    :return:
    """
    cat_sets = (set(to_cat_list(x)) for x in cat_series)
    return sorted(set.union(*cat_sets))
def get_english(cat):
    """
        将各个分类信息拆分为编码和英文名称
    :param cat:
    :return:
    """
    code, names = cat.split('.')
    if '|' in names:
        names = names.split('|')[1]
    return code, names.strip()
# print(get_english('2. Urgences logistiques | Vital Lines'))
all_cats = get_all_caegories(data.CATEGORY)
# 生成器表达式
english_mapping = dict(get_english(x) for x in all_cats)
print(english_mapping['2a'] + '\t/\t' +english_mapping['6c'])
def get_code(seq):
    """
        抽取出唯一的分类编码
    :param seq:
    :return:
    """
    return [x.split('.')[0] for x in seq if x]
all_codes = get_code(all_cats)
code_index = pd.Index(np.unique(all_codes))
# 构造全零DataFrame（列为分类编码，索引跟data的索引一样）
dummy_frame = pd.DataFrame(np.zeros((len(data), len(code_index))),
                           index=data.index,
                           columns=code_index)
# print(dummy_frame.ix[:, :6])
# 将各行中适当的项设置为1，然后再与data进行连接
for row, cat in zip(data.index, data.CATEGORY):
    codes = get_code(to_cat_list(cat))
    dummy_frame.ix[row, codes] = 1
data = data.join(dummy_frame.add_prefix('category_'))

def basic_haiti_map(ax=None, lllat=17.25, urlat=20.25, lllon=-75, urlon=-71):
    """
        绘制出一张简单的黑白海地地图
    :param ax:
    :param lllat:
    :param urlat:
    :param lllon:
    :param urlon:
    :return:
    """
    m = Basemap(ax=ax, projection='stere',
                lon_0=(urlon + lllon)/2,
                lat_0=(urlat + lllat)/2,
                llcrnrlat=lllat,
                urcrnrlat=urlat,
                llcrnrlon=lllon,
                urcrnrlon=urlon,
                resolution='f')
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    return m


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.subplots_adjust(hspace=0.05, wspace=0.05)
to_plot = ['2a', '1', '3c', '7a']
lllat = 17.25
urlat = 20.25
lllon = -75
urlon = -71
for code, ax in zip(to_plot, axes.flat):
    m = basic_haiti_map(ax, lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon)
    cat_data = data[data['category_%s' % code] == 1]
    x, y = m(cat_data.LONGITUDE.values, cat_data.LATITUDE.values)
    m.plot(x, y, 'k.', alpha=0.5)
    ax.set_title('%s : %s' % (code, english_mapping[code]))
plt.show()