"""
    使用决策树预测隐形眼镜类型
"""

import TP_Building_05 as TP5
import  TP_PlotTree_02 as TPPT

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = TP5.createTree(lenses,lensesLabels)
TPPT.createPlot(lensesTree)