'''
    python中使用Matplotlib注解绘制树形图
'''

import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

# 绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot():
   fig = plt.figure(1, facecolor='white')
   fig.clf()
   createPlot.ax1 = plt.subplot(111, frameon=False)
   plotNode(U'a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
   plotNode(U'a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
   plt.show()


# 测试
createPlot()
