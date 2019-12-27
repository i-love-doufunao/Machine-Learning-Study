import numpy as np


class LogisticRegression:
    def __init__(self):
        """
        初始化Logistics Regression模型
        """
        self.coef = None
        self.intercept_ = None
        self._theta = None

    """
    定义sigmoid方法
    y = 1 / 1 + e^-t
    参数：线性模型t
    输出：sigmoid表达式
    """

    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))


    """
    定义逻辑回归的损失函数
    参数：参数theta，构造好的矩阵X，标签y
    输出：损失函数表达式
    参见：逻辑回归的损失函数如下：J = - 1/m ...
    """
    def loss(self,X_b,y):


