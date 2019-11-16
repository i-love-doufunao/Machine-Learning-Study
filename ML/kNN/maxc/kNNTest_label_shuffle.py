from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from kNN.Metrics import accuracy_score


def train_test_split_by_concatenate(X, y, split_ratio):
    '''
    将X，y合并为矩阵并进行shuffle，再将shuffle后的数据按比例切分为测试集和训练集
    :param X:特征数据
    :param y:标签数据
    :param split_ratio:数据切分比例
    :return:特征训练数据集，特征测试数据集，标签训练数据集，便签测试数据集
    '''
    tempConcat = np.concatenate((X, y.reshape(-1, 1)), axis=1)
    np.random.shuffle(tempConcat)
    shuffle_X, shuffle_y = np.split(tempConcat, [4], axis=1)
    test_size = int(len(X) * split_ratio)
    X_train = shuffle_X[test_size:]
    y_train = shuffle_y[test_size:]
    X_test = shuffle_X[:test_size]
    y_test = shuffle_y[:test_size]
    return X_train, X_test, y_train, y_test


def train_test_split_by_shuffle_index(X, y, split_ratio):
    '''
    生成一个随机数数组（随机数小于特征数据长度），通过对该数组按比例切分得到对应的索引，根据索引找到X，y相应的数据
    :param X:特征数据
    :param y:标签数据
    :param split_ratio:数据切分比例
    :return:特征训练数据集，特征测试数据集，标签训练数据集，便签测试数据集
    '''
    shuffle_index = np.random.permutation(len(X))
    test_size = int(len(X) * split_ratio)
    test_index = shuffle_index[:test_size]
    train_index = shuffle_index[test_size:]
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    return X_train, X_test, y_train, y_test


iris = datasets.load_iris()
X = iris.data
print(X.shape)
y = iris.target
print(y.shape)

tempConcat = np.concatenate((X, y.reshape(-1, 1)), axis=1)
print(tempConcat)

print('-----shuffle------')

np.random.shuffle(tempConcat)
print(tempConcat)

# 参考split的描述，split是将数组按照中间[4]这个数组参数来进行拆分为多个子数组
shuffle_x, shuffle_y = np.split(tempConcat, [4], axis=1)

print(shuffle_x, shuffle_y)

X_train, X_test, y_train, y_test = train_test_split(shuffle_x, shuffle_x, random_state=2003)

print(X_train, X_test)

# print(accuracy_score(y_test))
