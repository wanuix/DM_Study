#-*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
from apriori import * #导入自行编写的apriori函数
"""
    Apriori关联规则算法

    关联规则通过量化的数字描述物品甲的出现对物品乙的出现有多大的影响

    它的模式属于描述型模式，发现关联规则的算法属于无监督学习的方法

    关联规则分析也是数据挖掘中最活跃的研究方法之一

    广泛运用于购物篮数据、生物信息学、医疗诊断、网页挖掘和科学数据分析中

    基本思想：
        通过链接产生候选项与其支持度，然后通过剪枝生成频繁项集

"""
"""
    Apriori算法的调用
"""


inputfile = 'menu_orders.xls'
# 结果文件
outputfile = 'apriori_rules.xls'


data = pd.read_excel(inputfile, header=None)


print(u'\n转换原始数据至0-1矩阵...')

# 转换0-1矩阵的过渡函数
ct = lambda x : pd.Series(1, index=x[pd.notnull(x)])

# 用map方式执行
# b = map(ct, data.as_matrix())
b = map(ct, data.values)

# 实现矩阵转换，空值用0填充
data = pd.DataFrame(list(b)).fillna(0)

print(u'\n转换完毕。')

# 删除中间变量b，节省内存
del b


# 最小支持度
support = 0.2
# 最小置信度
confidence = 0.5
# 连接符，默认'--'，用来区分不同元素，如A--B。需要保证原始表格中不含有该字符
ms = '---'
# 保存结果
find_rule(data, support, confidence, ms).to_excel(outputfile)