#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# In[2]:


from sklearn.model_selection import train_test_split #module for splitting datatset
from sklearn import metrics #module for evaluating performance


# In[3]:


#load your data
df = pd.read_csv("data.csv") #change the name accordingly
df.head() # prints top 5 rows from the datatset to check data is load or not


# In[4]:


# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)


# In[5]:


# remove duplicatesd
df = df.drop_duplicates()


# In[6]:


# prepare features
x = df.drop(['Label'],axis=1) #remove class or label
y = df['Label'] #load label


# In[7]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.25) #split datatset. Here ratio is 80:20. Change accordingly


# In[8]:


z = KNeighborsClassifier(n_neighbors=3) # KNN classifier for 3 neighbours
KNN = z.fit(x_train,y_train) # start training


# In[9]:


predict = KNN.predict(x_test) # performance in the test set


# In[10]:


print("Accuracy:", metrics.accuracy_score(y_test,predict)) # evaluating the performance based on accuracy
print("Precision:", metrics.precision_score(y_test, predict, average='weighted', zero_division=1))
print("Recall:", metrics.recall_score(y_test, predict, average='weighted'))
print("F1:", metrics.f1_score(y_test, predict, average='weighted'))


# In[11]:


# library for save and load scikit-learn models
import pickle
# file name, recommending *.pickle as a file extension
filename = "KNmodel.pickle"
# save model
pickle.dump(z, open(filename, "wb"))


# In[12]:


#Algorithm 2 - SVC with linear kernel
from sklearn.svm import SVC 
z_2 = SVC(kernel='linear')
SVM = z_2.fit(x_train,y_train) # start training
predict_2 = SVM.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test,predict_2))
print("Precision:", metrics.precision_score(y_test, predict_2, average='weighted', zero_division=1))
print("Recall:", metrics.recall_score(y_test, predict_2, average='weighted'))
print("F1:", metrics.f1_score(y_test, predict_2, average='weighted'))


# In[13]:


#Algorithm 2b - SVC with rbf kernel (applies to non-linear data)
from sklearn.svm import SVC 
z_2b = SVC(kernel='rbf')
SVM2b = z_2b.fit(x_train,y_train) # start training
predict_2b = SVM2b.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test,predict_2b))
print("Precision:", metrics.precision_score(y_test, predict_2b, average='weighted', zero_division=1))
print("Recall:", metrics.recall_score(y_test, predict_2b, average='weighted'))
print("F1:", metrics.f1_score(y_test, predict_2b, average='weighted'))


# In[14]:


import pickle
# file name, recommending *.pickle as a file extension
filename2 = "linearSVMmodel.pickle"
filename2b = "rbfSVMmodel.pickle"
# save model
pickle.dump(z_2, open(filename2, "wb"))
pickle.dump(z_2b, open(filename2b, "wb"))


# In[15]:


#Algorithm 3
from sklearn.ensemble import RandomForestClassifier
z_3 = RandomForestClassifier(n_estimators=100, random_state=42)
RFC = z_3.fit(x_train,y_train) # start training
predict_3 = RFC.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test,predict_3))
print("Precision:", metrics.precision_score(y_test, predict_3, average='weighted', zero_division=1))
print("Recall:", metrics.recall_score(y_test, predict_3, average='weighted'))
print("F1:", metrics.f1_score(y_test, predict_3, average='weighted'))

import pickle
# file name, recommending *.pickle as a file extension
filename3 = "KNmodel.pickle"
# save model
pickle.dump(z_3, open(filename3, "wb"))
# In[ ]:




