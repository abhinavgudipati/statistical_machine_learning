#!/usr/bin/env python
# coding: utf-8

# In[1]:


import PyPDF2
import os
import tabula
import nltk
import operator as op
import keras
import re
import string
import math
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import idx2numpy
import random as random
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical   
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[2]:


pwd


# In[3]:


os.chdir("/Users/abhinavgudipati/Desktop/SML_Assignment_03/")


# # Question 1 ( CIFAR-10 Dataset ) 

# # Question 1.1 :- Visualising 5 Samples from each Class

# In[4]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# In[5]:


pwd


# In[6]:


file_1 = '/Users/abhinavgudipati/Desktop/SML_Assignment_03/cifar-10-batches-py/data_batch_1'
file_2 = '/Users/abhinavgudipati/Desktop/SML_Assignment_03/cifar-10-batches-py/data_batch_2'
file_3 = '/Users/abhinavgudipati/Desktop/SML_Assignment_03/cifar-10-batches-py/data_batch_3'
file_4 = '/Users/abhinavgudipati/Desktop/SML_Assignment_03/cifar-10-batches-py/data_batch_4'
file_5 = '/Users/abhinavgudipati/Desktop/SML_Assignment_03/cifar-10-batches-py/data_batch_5'
test_file = '/Users/abhinavgudipati/Desktop/SML_Assignment_03/cifar-10-batches-py/test_batch'


# In[7]:


data_batch_1 = unpickle(file_1)
data_batch_2 = unpickle(file_2)
data_batch_3 = unpickle(file_3)
data_batch_4 = unpickle(file_4)
data_batch_5 = unpickle(file_5)
test_batch = unpickle(test_file)


# In[8]:


test_batch.keys()


# In[9]:


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


# # Visualising 5 Images from each class

# In[11]:


a = 0
classLabel = 0
new =0
b = 10 
d = 10
plt.figure(figsize=(b,d))

for i in range(50):
  for j in range(new,len(y_train)):
    if op.eq(y_train[j],a):
      plt.subplot(10,5,op.add(i,1))
      plt.imshow(x_train[j])
      new = op.add(j,1)
      if op.lt(classLabel,4):
        classLabel= op.iadd(classLabel,1)
        break;
      else:
        classLabel=0;
        a= op.iadd(a,1)
        break;
plt.show()        


# # Question 1.2

# In[12]:


(X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()


# In[13]:


X_train = X_train.reshape(50000,3*32*32)
X_test = X_test.reshape(10000,3*32*32)


# In[14]:


def reports():
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, Y_train)
    y_pred = lda.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test, y_pred)
    entireReport = classification_report(Y_test,y_pred)
    confusionMat = confusion_matrix(Y_test,y_pred)
    accu = confusionMat.diagonal()/confusionMat.sum(axis=1)
    points = [0,1,2,3,4,5,6,7,8,9]
    for i in range(10):
        print(f'Accuracy of class {i} is {accu[i]}')
    print(accu)
    print("Overall Accuracy:",accuracy)
    print(entireReport)


# In[15]:


reports()


# In[ ]:





# In[ ]:




