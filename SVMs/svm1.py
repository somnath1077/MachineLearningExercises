from sklearn import svm
import numpy as np

X = np.array([[0, 1], [1, 1]])
y = np.array([1, -1])

clf = svm.SVC()
clf.fit(X, y)
print(clf.predict([[1, 1]]))