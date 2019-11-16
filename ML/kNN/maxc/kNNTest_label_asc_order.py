from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X.size, y.size)

# random_state是一个seed；也可以设置test_size=0.3,随机70%训练，30%测试
# 千万注意不要写错了返回值
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4798)
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)

"""
这部分的代码主要用来做预测以及计算准确率。
计算准确率的逻辑也很简单，就是判断预测和实际值有多少是相等的。
如果相等则算预测正确，否则预测失败。
"""
correct = np.count_nonzero((knn_clf.predict(X_test) == y_test) == True)
print("Accuracy is:%.3f" % (correct / len(X_test)))

print("finding correct parameter")

best_method = ""
best_score = 0.0
best_k = -1
for method in ["uniform", "distance"]:
    for k in range(1, 11):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights=method, p=2)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_k = k
            best_score = score
            best_method = method
print("best_method = ", method)
print("best_k = ", best_k)
print("best_score = ", best_score)

print("finding correct parameter by param search")

# 首先是一个数组；数组的每个元素是一个字典。即{"weights": ["uniform"], "n_neighbors": [i for i in range(1, 11)]}
# 是一个字典
param_search = [
    {"weights": ["uniform"], "n_neighbors": [i for i in range(1, 11)]
     },
    {"weights": ["distance"],
     "n_neighbors": [i for i in range(1, 11)],
     "p": [i for i in range(1, 6)]
     }
]

# knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_search)
best_kNN_clf = grid_search.estimator
print(grid_search)
print(grid_search.estimator)
# print(grid_search.best_score_)
