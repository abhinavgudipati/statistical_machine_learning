#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import operator as op
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


# In[2]:


pwd


# In[3]:


os.chdir("/Users/abhinavgudipati/Desktop/SML_Assignment_03/")


# In[4]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2])) 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))


# In[5]:


X_train.shape


# In[6]:


pca = PCA(n_components = 15)
pca.fit(X_train)
x_train = pca.transform(X_train)
x_test = pca.transform(X_test)


# In[7]:


def fuctionFDA(X, y):
    N, D = X.shape
    St = op.matmul((op.sub(X,np.mean(X))).T,(op.sub(X,np.mean(X))))
    print(St.shape)
    Sw = np.zeros((D, D))
    classes = op.add(np.max(y),1)
    for c in range(classes):
        Xc = X[op.eq(y,c), :]
        Sw = op.iadd(Sw,op.matmul((op.sub(Xc , np.mean(Xc))).T ,(op.sub(Xc , np.mean(Xc)))))
    Sw = op.iadd(Sw,np.dot((op.sub(Xc , np.mean(Xc))).T, (op.sub(Xc,np.mean(Xc)))))
    print(Sw.shape)
    w, v = np.linalg.eigh(op.matmul(np.linalg.inv(Sw),(op.sub(St,Sw))))
    W = [np.array(x) for _, x in sorted(zip(w, v), reverse=True)]
    return np.array(W[ : op.sub(classes,1)])


# In[8]:


L = fuctionFDA(x_train,y_train)


# In[9]:


Y_train = op.matmul(x_train,L.T)
Y_test = op.matmul(x_test,L.T)
Y_train.shape


# In[10]:


# part c
lda = LinearDiscriminantAnalysis()
lda.fit(Y_train, y_train)


# In[11]:


lda.score(Y_test, y_test)


# In[12]:


classes = op.add(np.max(y_train),1)
glasses, total = op.mul([0],classes), op.mul([0],classes)
y_predicted = lda.predict(Y_test)
for i in range(y_test.shape[0]):
    glasses[y_test[i]] = op.iadd(glasses[y_test[i]],int(op.eq(y_test[i],y_predicted[i])))
    total[y_test[i]] = op.iadd(total[y_test[i]],1)


# In[13]:


def classAccuracy(classes):
    for i in range(op.add(np.max(y_train),1)):
        ans = glasses[i]/total[i]
        print(f'Accuracy of class {i} is {ans}')


# In[14]:


classAccuracy(classes)


# In[ ]:




