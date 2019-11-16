from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import operator as opt


def euc_dist(instance1, instance2):
    dist = (instance1 - instance2) ** 2
    dist = dist.sum(axis=1) ** 0.5
    return dist


def knn_classify(X, y, testInstance, k):
    sortedDist = euc_dist(X, testInstance).argsort()
    indices = sortedDist[:k]
    labelCount = {}
    for i in indices:
        label = y[i]
        labelCount[label] = labelCount.get(label, 0) + 1
    sortedCount = sorted(labelCount.items(), key=opt.itemgetter(1), reverse=True)
    return sortedCount[0][0]


"""
1.把一个物体表示成向量
2.标记好物品的标签
3.计算两个物体相似度/距离
4.选择合适的K

4类特征，0，1，2 等3种标签
"""

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X.size, y.size)

"""
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [6.2 3.4 5.4 2.3]
 [5.9 3.  5.1 1.8]] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
"""

# random_state是一个seed；也可以设置test_size=0.3,随机70%训练，30%测试
# 千万注意不要写错了返回值
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2003)

"""
这部分的代码主要用来做预测以及计算准确率。
计算准确率的逻辑也很简单，就是判断预测和实际值有多少是相等的。
如果相等则算预测正确，否则预测失败。
"""
result = [knn_classify(X_train, y_train, test_data, 3) for test_data in X_test]
print(result)
# print("Accuracy is:%.3f" % (correct / len(X_test)))
