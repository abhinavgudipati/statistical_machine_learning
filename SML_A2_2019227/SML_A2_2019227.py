# -*- coding: utf-8 -*-
"""Abhinav.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZY8nHZrmid_U7rZ3wprdT_VdQgf4uQLh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from sklearn.model_selection import train_test_split
import seaborn as sns

np.random.seed(0)

def create_data(p,size):
    return bernoulli.rvs(p,size=size)

x1,x2=create_data(0.5,100),create_data(0.8,100)
df1=pd.DataFrame({"x1":x1,"x2":x2})
df1.head()

x1,x2 = create_data(0.9,100), create_data(0.2,100)
df2=pd.DataFrame({"x1":x1,"x2":x2})
df2.head()

df = pd.concat([df1, df2])
df.head(),df.shape

a = 0.5
train_df1, test_df1 = train_test_split(df1, test_size = a)
train_df2, test_df2 = train_test_split(df2, test_size = a)

def do_something(df):
    res =[]
    i = 1 
    while(i<51):
        temp = df.sample(i)
        temp.columns = ['xx','yy']
        temp_prediction =np.array([np.sum(temp['xx'])/i, np.sum(temp['yy'])/i])
        res.append(temp_prediction)
        i = i + 1 
    res=np.array(res)
    return res

res=do_something(df1)
mle1 = np.average(res, axis=0)
res,mle1

pd.DataFrame(res[:,0]).plot(),pd.DataFrame(res[:,1]).plot()

res=do_something(df1)
mle2 = np.average(res, axis=0)
res,mle2

pd.DataFrame(res[:,0]).plot(),pd.DataFrame(res[:,1]).plot()

sns.scatterplot(train_df1['x1'], train_df1['x2'])

def derivative(mu, x):
    return (mu**x)*((1-mu)**x)

def do_something2(mle1,mle2,test_df):
    temp1 = list(derivative(mle1[0*1+0], test_df['x1'])*derivative(mle1[1], test_df['x1']))
    temp2 = list(derivative(mle2[0*1+0], test_df['x1'])*derivative(mle2[1], test_df['x2']))
    cat = []
    i = 1 
    while(i<len(temp1)):
        if temp1[i]>temp2[i]:
          cat.append(0*1+0)
        else:
          cat.append(1*1+0)
        i = i + 1*1
    return np.array(cat)

do_something2(mle1,mle2,test_df1)

do_something2(mle1,mle2,test_df2)
