import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [2, 1], [3, 3,], [1, 1], [2, 1], [3, 2]])

Y = np.array([1, 1, 1, 2, 2, 2])

clf = GaussianNB()
clf.fit(X, Y)

print(clf.predict([[-0.8, 1]]))


## from sklearn.naive_bayes import GaussianNB
## clf = GaussianNB()
## clf.fit(features_train, labels_train)
## pred = clf.predict(features_test)