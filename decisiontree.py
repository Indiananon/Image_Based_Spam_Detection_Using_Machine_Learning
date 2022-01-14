# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 14:25:25 2018

@author: Ankit
"""

# Decision Tree Classificastion

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('trained.csv')
X = dataset.iloc[:, [0,2,4,6,8,10,12,14,16,18,20]].values
dataset.isnull().sum()
#X = dataset.iloc[:, [2,3,5,6,7,8,9,14,17,18,19,20]].values

y = dataset.iloc[:, -1:].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.9, random_state = 0)


"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

"""


X.dtype
y.dtype
X_train.dtype
np.where(np.isnan(X))
np.where(np.isnan(y))

np.isfinite(X)
np.isfinite(y)
np.isfinite(X_train)

# Fitting GradientBoostingClassifier to the dataset
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state = 78)
classifier.fit(X, y)

# Predicting a new result
y_pred = classifier.predict(X_test)


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
