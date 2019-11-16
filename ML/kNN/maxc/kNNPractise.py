import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter

raw_data_x = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343853454, 3.368312451],
              [3.582294121, 4.679917921],
              [2.280362211, 2.866990212],
              [7.423436752, 4.685324231],
              [5.745231231, 3.532131321],
              [9.172112222, 2.511113104],
              [7.927841231, 3.421455345],
              [7.939831414, 0.791631213]]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
X_train = np.array(raw_data_x)
Y_train = np.array(raw_data_y)

# #绘图
# plt.scatter(X_train[Y_train == 0, 0], X_train[Y_train == 0, 1], color='g', label='Tumor Size')
# plt.scatter(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1], color='r', label='Time')
#
# plt.xlabel('Tumor Size')
# plt.ylabel('Time')
# plt.axis([0, 10, 0, 5])
# plt.show()

x_test = [8.90933607318, 3.365731514]
distances = [sqrt(np.sum((x_train - x_test) ** 2)) for x_train in X_train]

nearest = np.argsort(distances)
k = 6

# 最近邻k个标签值
topK_y = [Y_train[i] for i in nearest[:6]]

#选频次最多的标签
votes=Counter(topK_y).most_common(2)
print(votes)




