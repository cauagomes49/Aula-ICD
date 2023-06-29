import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt


iris = pd.read_csv('iris.data', header=None)
iris = iris.to_numpy()

X = iris[:,:-1]
y = iris[:, -1]

print(X.shape)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

clf.predict([[2,2,3,1]])

y_estimado =[]
for flor in X:
    y_estimado.append(clf.predict([flor])[0])

print(y_estimado ==y)
