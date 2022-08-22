import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

irisDataSet = load_iris()
X_train, X_test, y_train, y_test = train_test_split(irisDataSet.data, irisDataSet.target, random_state=0)

model = DecisionTreeClassifier()
modelTest = model.fit(X_train, y_train)
X_New = np.array([[6.0, 3.23, 4.5, 2.0]])
print(modelTest.predict(X_New))
