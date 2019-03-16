"""
    使用Matplotliv创建散点图
"""

import matplotlib
import matplotlib.pyplot as plt
import knn_DatingWebsiteMatching_02
from numpy import *


datingDatMat = knn_DatingWebsiteMatching_02.datingDataMat
datingLabels = knn_DatingWebsiteMatching_02.datingLabels

fig1 = plt.figure()
#子图未知，第一个数字是指行数，第二个数字是指列数，第三个数字是子图的位置
ax1 = fig1.add_subplot(111)
#
ax1.scatter(datingDatMat[:,0], datingDatMat[:,1],
           15.0 * array(datingLabels), 15.0 * array(datingLabels))
handles, labels = plt.gca().get_legend_handles_labels()
# plt.legend(datingLabels)

# fig2 = plt.figure()
# ax2 = plt.subplot(122)
# ax2.scatter(datingDatMat[:,1], datingDatMat[:,2],
#            15.0 * array(datingLabels), 15.0 * array(datingLabels))
#
# ax2.legend(['1', '2', '3',], bbox_to_anchor=(0., 1.02, 1., .102), loc=1,
#        ncol=3, mode="expand", borderaxespad=0.)

plt.show()