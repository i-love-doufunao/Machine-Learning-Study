import numpy as np
from sklearn.neighbors import KNeighborsClassifier

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

x_test = [8.90933607318, 3.365731514]

kNN_classifier = KNeighborsClassifier(n_neighbors=6)
kNN_classifier.fit(X_train, Y_train)
y_predict = kNN_classifier.predict(np.array(x_test).reshape(1, -1))
print(y_predict)