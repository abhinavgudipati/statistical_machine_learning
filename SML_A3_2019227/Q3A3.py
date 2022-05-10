#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import operator as op
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[2]:


pwd


# In[3]:


os.chdir("/Users/abhinavgudipati/Desktop/SML_Assignment_03/fminst/")


# In[4]:


training_Data = pd.read_csv('fashion-mnist_train.csv')
testing_Data = pd.read_csv('fashion-mnist_test.csv')
training_Data = training_Data.sort_values(by='label')
testing_Data = testing_Data.sort_values(by='label')


# In[5]:


training_Data


# In[6]:


#dropping the label column and converting it into a numpy array 
X_train = training_Data.drop(['label'],axis=1).to_numpy()
X_train


# In[7]:


X_train.shape


# In[8]:


#converting Y_train into a numpy array 
y_train = training_Data['label'].to_numpy()
y_train.shape


# In[9]:


X_test = testing_Data.drop(['label'],axis=1).to_numpy()
X_test.shape


# In[10]:


y_test = testing_Data['label'].to_numpy()
y_test.shape


# In[11]:


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


# In[12]:


L = fuctionFDA(X_train,y_train)


# In[13]:


Y_train = op.matmul(X_train,L.T)
Y_test = op.matmul(X_test,L.T)
Y_train.shape


# In[14]:


lda = LinearDiscriminantAnalysis()
lda.fit(Y_train, y_train)


# In[15]:


lda.score(Y_test, y_test)


# In[16]:


classes = op.add(np.max(y_train),1)
glasses, total = op.mul([0],classes), op.mul([0],classes)
y_predicted = lda.predict(Y_test)
for i in range(y_test.shape[0]):
    glasses[y_test[i]] = op.iadd(glasses[y_test[i]],int(op.eq(y_test[i],y_predicted[i])))
    total[y_test[i]] = op.iadd(total[y_test[i]],1)


# In[17]:


def classAccuracy(classes):
    for i in range(op.add(np.max(y_train),1)):
        ans = glasses[i]/total[i]
        print(f'Accuracy of class {i} is {ans}')


# In[18]:


classAccuracy(classes)


# In[ ]:




