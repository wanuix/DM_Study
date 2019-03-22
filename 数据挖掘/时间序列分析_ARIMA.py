#-*- coding: utf-8 -*-
from __future__ import print_function
"""
    Arima时序模型
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from statsmodels.graphics.api import qqplot
from statsmodels.graphics.tsaplots import plot_acf  # 自相关图
from statsmodels.tsa.stattools import adfuller as ADF  # 平稳性检测
from statsmodels.graphics.tsaplots import plot_pacf  # 偏自相关图
from statsmodels.stats.diagnostic import acorr_ljungbox     # 白噪声检验


# 参数初始化
# 读取数据，指定日期列为指标，Pandas自动将“日期”列识别为Datetime格式
discfile = 'arima_data.xls'
data = pd.read_excel(discfile, index_col=0)
print(data.head())
print('\n Data Types:')
print(data.dtypes)


# =================================得到平稳的时间序列==================================
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 时序图
data.plot()
plt.show()
# 自相关图
plot_acf(data).show()
# 平稳性检测
print(u'原始序列的ADF检验结果为：', ADF(data[u'销量']))
# 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore
# 若pvalue显著大于0.05，则该序列为非平稳序列
"""
    该数据的自相关图
    在4阶后才落入区间内，而且自相关系数长期大于0，显示出很强的自相关性
    从平稳性检验结果可以看出，pvalue显著大于0.05，该序列为非平稳序列
"""


"""
分析：判断平稳与否的话，用自相关图和偏相关图就可以了。
平稳的序列的自相关图和偏相关图不是拖尾就是截尾。
截尾就是在某阶之后，系数都为 0 
怎么理解呢，看上面偏相关的图，当阶数为 1 的时候，系数值还是很大， 0.914.
二阶长的时候突然就变成了 0.050. 后面的值都很小，认为是趋于 0 ，这种状况就是截尾。
再就是拖尾，拖尾就是有一个衰减的趋势，但是不都为 0 。
自相关图既不是拖尾也不是截尾。以上的图的自相关是一个三角对称的形式，这种趋势是单调趋势的典型图形。
 
下面是通过自相关的其他功能
如果自相关是拖尾，偏相关截尾，则用 AR 算法
如果自相关截尾，偏相关拖尾，则用 MA 算法
如果自相关和偏相关都是拖尾，则用 ARMA 算法， ARIMA 是 ARMA 算法的扩展版，用法类似 。

不平稳，怎么办？
答案是差分，什么是差分？不介绍了，给个链接：
http://zh.wikipedia.org/wiki/%E5%B7%AE%E5%88%86
还是上面那个序列，两种方法都证明他是不靠谱的，不平稳的。
确定不平稳后，依次进行1阶、2阶、3阶...差分，直到平稳位置。


观察自相关图与偏相关图最主要的目的还是确定序列的ARMA（p，q）模型的具体形式。

首先，需要明确这样几对概念：
第一，自回归过程与移动平均过程。
    自回归由序列的滞后变量的线性组合以及白声噪（符合0均值固定方差的随机干扰项）相加而成
    移动平均过程为白声噪的线性组合构成；
第二，拖尾和截尾。
这一对概念从图表上很容易看出
前者指AC或者PAC呈几何衰减（指数式衰减或者正弦式衰减）
后者指AC或者PAC在某一阶之前明显不为0，之后突然接近或者等于0.
其实，从字面上也很好理解，拖尾就是拖拖拉拉，截尾就是抽刀断水。


其次是对ARMA模型的分解。

AR(p)模型，从自相关函数ACF来看，在自回归方程的基础上可以很简单地构造自相关系数，
最后发现自相关系数等于w^k（w为自回归系数）,
对于平稳时间序列（注意这一前提条件，如果放开这一条件图形将会很难识别），|w|<1，
所以当w>0时，ACF呈现为指数式衰减至0。
当w<0时，ACF则正负交替呈指数衰减至0，整体表现则是正弦式衰减；
从偏相关函数PACF来看，这就相当明显了，
因为PACF与自回归方程的形式完全一样，只是自回归方程只有滞后p期，而PACF则有更多的滞后项。
于是乎，很明显，当k<=p,偏相关系数不等于0，当k>p，偏相关系数等于0，明显呈现出截尾现象。

MA（q）模型，从自相关函数ACF来看，在移动平均方程的基础上也可以很简单地构造自相关系数，
这时候的自相关函数为分段函数，
当k<=q,偏相关系数不等于0，
当k>q，偏相关系数等于0，明显呈现出截尾现象；
从偏相关函数PACF来看，
任何一个可逆的MA（q）过程
都可以转换成一个无限阶、系数按几何衰减的AR过程（将白噪声替换为序列的滞后形式即可），
呈现拖尾现象。
与AR（p）不同的是，当v>0（v为移动平均系数）时，PACF呈现为交替式正弦衰减。
当v<0时，PACF则呈指数衰减至0。

ARMA（p，q）模型则是两者的结合，实际判别p、q值时还是比较依赖经验的。
"""


"""
ARIMA模型对时间序列的要求是平稳的。

因此，当你得到一个非平稳的时间序列时
首先要做的即是做时间序列的差分，直到得到一个平稳时间序列。  
    
平稳：就是围绕着一个常数上下波动且波动范围有限，即有常数均值和常数方差。
如果有明显的趋势或周期性，那它通常不是平稳序列。一般有三种方法：
（1）直接画出时间序列的趋势图，看趋势判断。
（2）画自相关和偏自相关图：平稳的序列的自相关图（Autocorrelation）
        和偏相关图（Partial Correlation）要么拖尾，要么是截尾。
（3）单位根检验：检验序列中是否存在单位根，如果存在单位根就是非平稳时间序列。

不平稳序列可以通过差分转换为平稳序列。
d阶差分就是相距d期的两个序列值之间相减。
如果一个时间序列经过差分运算后具有平稳性，则该序列为差分平稳序列，
可以使用ARIMA模型进行分析。

如果你对时间序列做d次差分才能得到一个平稳序列
那么可以使用ARIMA(p,d,q)模型，其中d是差分次数。
"""


# 差分后的时序图
# 差分：1阶差分是下一时刻第数据减当前时刻第数据,d阶差分就是相距d期的两个序列值之间相减。
D_data = data.diff().dropna()
D_data.columns = [u'销量差分']
# 时序图
D_data.plot()
plt.show()
# 自相关图
plot_acf(D_data).show()
# 偏自相关图
plot_pacf(D_data).show()
# 平稳性检测
print(u'差分序列的ADF检验结果为：', ADF(D_data[u'销量差分']))
# 白噪声检验
print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))
"""
    该数据对应的差分后的数据D_data
    差分后的学列迅速落入区间内，并呈现出向0靠拢的趋势
    序列没有自相关性
    通过偏自相关图可以看出，差分后的序列也没有显示出偏自相关性
    从平稳性检验结果可以看出，pvalue小于0.05，该序列为平稳序列
    通过白噪声检验结果可以看出，检验的p值小于0。05，通过白噪声检验，序列为白噪声序列
"""
# 返回统计量和p值

# 一阶差分
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
diff1 = data.diff(1)
diff1.plot(ax=ax1)

# 二阶差分
fig = plt.figure(figsize=(12, 8))
ax2= fig.add_subplot(111)
diff2 = data.diff(2)
diff2.plot(ax=ax2)


# ================================选择合适的ARIMA模型=================================
# 合适的p, q
# 因为二阶差分后的时间序列与一阶差分相差不大，
# 且二者随着时间推移，
# 时间序列的均值和方差保持不变，
# 因此可以将差分次数d设为1
dta = data.diff(1)[1:]  # dta.diff(1) 以一阶差分序列作为平稳时间序列
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig1 = sm.graphics.tsa.plot_acf(dta[u'销量'], lags=10, ax=ax1)    # lags表示滞后的阶数
ax2 = fig.add_subplot(212)
fig2 = sm.graphics.tsa.plot_pacf(dta[u'销量'], lags=10, ax=ax2)
"""
通过观察acf图和pacf图，可以得到：
* 自相关图显示滞后有2个阶超出了置信边界（第一条线代表起始点，不在滞后范围内）；
* 偏相关图显示在滞后1至2阶（lags 1,2）时的偏自相关系数超出了置信边界，
    从lag 2之后偏自相关系数值缩小至0
则有以下模型可以供选择：
1. ARMA(0,2)模型：即自相关图在滞后2阶之后缩小为0，且偏自相关缩小至0，
    则是一个阶数q=2的移动平均模型；
2. ARMA(1,0)模型：即偏自相关图在滞后1阶之后缩小为0，且自相关缩小至0，
    则是一个阶层p=7的自回归模型；
3. ARMA(0,1)模型：即自相关图在滞后1阶后缩小为0，且偏自相关缩小至0，
    则是一个阶数q=1的自回归模型
4. …其他供选择的模型。

补充：https://www.jianshu.com/p/9a05472b0e7d
"""
# 模型
arma_mod20 = sm.tsa.ARMA(dta, (2, 0)).fit()     # ARMA(0,2)
print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)
arma_mod01 = sm.tsa.ARMA(dta, (0, 1)).fit()     # ARMA(1,0)
print(arma_mod01.aic, arma_mod01.bic, arma_mod01.hqic)
arma_mod10 = sm.tsa.ARMA(dta, (1, 0)).fit()     # ARMA(0,1)
print(arma_mod10.aic, arma_mod10.bic, arma_mod10.hqic)
"""
    AIC 赤池信息量，鼓励数据拟合的优良性，优先考虑模型应是AIC值最小的一个
    BIC 贝叶斯信息量
    HQ  Hannan-Quinn Criterion
    三个参数最小的模型，为最佳模型
"""


# ================================模型检验====================================
# 残差QQ图
# 用于验证一组数据是否满足某个分布，或验证某两组数据是否来自同一分布
resid = arma_mod01.resid
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)

# 残差自相关检验
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(arma_mod01.resid.values.squeeze(), lags=10, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(arma_mod01.resid, lags=10, ax=ax2)


# D-W检验
"""
德宾-沃森（Durbin-Watson）检验。
德宾-沃森检验,简称D-W检验，是目前检验自相关性最常用的方法，
但它只使用于检验一阶自相关性。
因为自相关系数ρ的值介于-1和1之间，所以 0≤DW≤４。
并且
    DW＝O＝＞ρ＝１　　  即存在正自相关性
    DW＝４＜＝＞ρ＝－１　即存在负自相关性
    DW＝２＜＝＞ρ＝０　　即不存在（一阶）自相关性
因此
当DW值显著的接近于O或４时，则存在自相关性，
而接近于２时，则不存在（一阶）自相关性。

这样只要知道ＤＷ统计量的概率分布，在给定的显著水平下，
根据临界值的位置就可以对原假设Ｈ０进行检验。

sm.stats.durbin_watson(arma_mod20.resid.values)
"""
print(sm.stats.durbin_watson(arma_mod01.resid.values))
"""
    检验结果是1.9734866184998665 说明不存在相关性
"""


# Ljung-Box检验
"""
Ljung-Box检验是对随机性的检验,或者说是对时间序列是否存在滞后相关的一种统计检验。　
说明　
对于滞后相关的检验，我们常常采用的方法还包括计算ACF和PCAF并观察其图像，
但是无论是ACF还是PACF都仅仅考虑是否存在某一特定滞后阶数的相关。
LB检验则是基于一系列滞后阶数，判断序列总体的相关性或者说随机性是否存在

时间序列中最基本的模型就是高斯白噪声序列
而对于ARIMA模型，其残差被假定为高斯白噪声序列
所以当我们用ARIMA模型去拟合数据时
拟合后我们要对残差的估计序列进行LB检验，判断其是否是高斯白噪声
如果不是，那么说明该ARIMA模型也许并不是一个适合样本模型
"""
r, q, p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
datap = np.c_[range(1, 36), r[1:], q, p]
table = pd.DataFrame(datap, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))
"""
    检验的结果就是看最后一列前12行的检验概率（一般观察滞后1～12阶）
    如果检验概率小于给定的显著水平，比如0.05、0.10等
    就是拒绝原假设
    其原假设是相关系数为0
    
    就结果来看，显著水平为0.05，那么相关系数与0没有显著差异
    即为白噪声序列
"""


# ================================模型预测====================================
# 预测
predict_sunspots = arma_mod01.predict('2015-2-07', '2015-2-15', dynamic=True)
fig, ax = plt.subplots(figsize=(12, 8))
print(predict_sunspots)
predict_sunspots[0] += data['2015-02-06':][u'销量']
data = pd.DataFrame(data)
for i in range(len(predict_sunspots)-1):
    predict_sunspots[i+1] = predict_sunspots[i] + predict_sunspots[i+1]
print(predict_sunspots)
ax = data.ix['2015':].plot(ax=ax)
predict_sunspots.plot(ax=ax)
plt.show()
