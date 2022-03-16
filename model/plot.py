

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# d=pd.read_excel('E:\Python\projects\data\data100.xlsx',header=None)
# d=d[0]
# d=list(d)

ages = range(11)
count = [0.1, 0.3, 0.2, 0.9, 0.8, 0.9, 0.1, 0.7, 0.8, 0.8, 0.6]
plt.bar(ages, count)
# params
# x: 条形图x轴
# y：条形图的高度
# width：条形图的宽度 默认是0.8
# bottom：条形底部的y坐标值 默认是0
# align：center / edge 条形图是否以x轴坐标为中心点或者是以x轴坐标为边缘
plt.legend()




plt.show()


