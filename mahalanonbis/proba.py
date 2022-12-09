import heapq
import numpy as np
from numpy import dot
from collections import Counter
from operator import itemgetter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()
X = iris.data  # we only take the first two features.
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


test = list(zip(X_test, y_test))

for i in range(len(test)):
    print(test[i])