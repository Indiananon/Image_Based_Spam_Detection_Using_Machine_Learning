# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 14:26:44 2018

@author: Ankit
"""


# coding: utf-8

# In[61]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
import itertools
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[40]:


dataset = pd.read_csv('trained.csv')
X = dataset.iloc[:, [1,2,4,5,7,8,9,18,20]].values
dataset.isnull().sum()
#X = dataset.iloc[:, [0,2,3,5,6,7,8,9,11,12,16,18,19,20]].values

y = dataset.iloc[:, -1:].values

# In[43]:


print(X.shape)
print(y.shape)


# In[44]:


np.where(np.isnan(y))


# In[45]:




# In[46]:



# In[47]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



X.dtype
y.dtype
X_train.dtype
np.where(np.isnan(X))
np.where(np.isnan(y))

np.isfinite(X)
np.isfinite(y)
np.isfinite(X_train)

# In[63]:


classifier = RandomForestClassifier(n_estimators = 10, random_state = 0)
classifier.fit(X_train, np.ravel(y_train))


# In[64]:


y_pred = classifier.predict(X_test)
y_pred


# In[65]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)





#test
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)






