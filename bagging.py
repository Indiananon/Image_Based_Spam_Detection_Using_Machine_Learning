# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 14:15:35 2018

@author: Ankit
"""

# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('trained.csv')
X = dataset.iloc[:, [0,2,4,6,8,10,12,14,16,18,20]].values
dataset.isnull().sum()
#X = dataset.iloc[:, [2,3,5,6,7,8,9,14,17,18,19,20]].values

y = dataset.iloc[:, -1:].values

y=y.reshape(-1,1)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.ensemble import BaggingClassifier
classifier = BaggingClassifier()
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)



# Predicting the Test set results

dataset = pd.read_csv('check1.csv')
dataset.isnull().sum()
X_test = dataset.iloc[:, [2,3,5,6,7,8,9,14,17,18,19,20]].values

y_pred = classifier.predict(X_test)

print(y_pred)
# Making the Confusion Matrix

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)



from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

