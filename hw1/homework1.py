#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import linear_model
import numpy
import random
import gzip
import math


# In[2]:


def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[3]:


f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))


# In[4]:


len(dataset)


# In[5]:


answers = {} # Put your answers to each question in this dictionary


# In[6]:


dataset[0]


# In[ ]:


### Question 1


# In[15]:


def feature(datum):
    return [1, datum['review_text'].count('!')]


# In[22]:


X = [feature(data) for data in dataset]
Y = [data['rating'] for data in dataset ]
theta,residuals,rank,s = numpy.linalg.lstsq(X, Y, rcond=None)
theta0 = theta[0]
theta1 = theta[1]
Y_pred = numpy.dot(X,theta)
mse = numpy.square(numpy.subtract(Y,Y_pred)).mean()


# In[27]:


answers['Q1'] = [theta0, theta1, mse]
print(answers['Q1'] )


# In[28]:


assertFloatList(answers['Q1'], 3) # Check the format of your answer (three floats)


# In[29]:


### Question 2


# In[30]:


def feature(datum):
    return [1, len(datum['review_text']), datum['review_text'].count('!')]
    


# In[88]:


X = [feature(data) for data in dataset]
Y = [data['rating'] for data in dataset ]
theta,residuals,rank,s = numpy.linalg.lstsq(X, Y, rcond=None)
theta0 = theta[0]
theta1 = theta[1]
theta2 = theta[2]
print(theta)
Y_pred = numpy.dot(X,theta)
mse = numpy.square(numpy.subtract(Y,Y_pred)).mean()


# In[34]:


answers['Q2'] = [theta0, theta1, theta2, mse]


# In[56]:


assertFloatList(answers['Q2'], 4)


# In[57]:


### Question 3


# In[66]:


def feature(datum, deg):
    return [datum['review_text'].count('!') ** i for i in range(deg+1)]

mses = []
for i in range(1, 6):
    X = [feature(data,i) for data in dataset]
    Y = [data['rating'] for data in dataset ]
    theta,residuals,rank,s = numpy.linalg.lstsq(X, Y, rcond=None)
    Y_pred = numpy.dot(X,theta)
    mses.append(numpy.square(numpy.subtract(Y,Y_pred)).mean())
    


# In[67]:


answers['Q3'] = mses
print(answers['Q3'])


# In[ ]:


assertFloatList(answers['Q3'], 5)# List of length 5


# In[92]:


### Question 4

training_data = dataset[:len(dataset)//2]
test_data = dataset[(len(dataset)//2) :]


# In[95]:



def feature(datum, deg):
    return [datum['review_text'].count('!') ** i for i in range(deg+1)]

mses = []
for i in range(1, 6):
    X_train = [feature(data,i) for data in training_data]
    Y_train = [data['rating'] for data in training_data]
    
    theta,residuals,rank,s = numpy.linalg.lstsq(X_train, Y_train, rcond=None)
    
    X_test = [feature(data, i) for data in test_data]
    Y_test = [data['rating'] for data in test_data]

    Y_pred = numpy.dot(X_test,theta)
    mses.append(numpy.square(numpy.subtract(Y_test,Y_pred)).mean())


# In[96]:


answers['Q4'] = mses
print(answers['Q4'])


# In[ ]:


assertFloatList(answers['Q4'], 5)


# In[101]:


### Question 5


# In[108]:


def feature(datum):
    return [1]

X_train = [feature(data) for data in training_data]
Y_train = [data['rating'] for data in training_data]

theta,residuals,rank,s = numpy.linalg.lstsq(X_train, Y_train, rcond=None)

X_test = [feature(data) for data in test_data]
Y_test = [data['rating'] for data in test_data]

Y_pred = numpy.dot(X_test,theta)
mae = numpy.absolute (numpy.subtract(Y_test,Y_pred)).mean()


# In[109]:


answers['Q5'] = mae
print(mae)


# In[107]:


assertFloat(answers['Q5'])


# In[256]:


### Question 6


# In[257]:


f = open("beer_50000.json")
dataset = []
for l in f:
    if 'user/gender' in l:
        dataset.append(eval(l))


# In[272]:


len(dataset)


# In[275]:


X = [[1,data['review/text'].count('!')] for data in dataset]
Y = [data['user/gender'] == 'Female' for data in dataset]





# In[279]:


mod = linear_model.LogisticRegression(C=1.0)
mod.fit(X, Y)

Y_pred = mod.predict(X)

TP = sum(numpy.logical_and(Y_pred, Y))
FP = sum(numpy.logical_and(Y_pred, numpy.logical_not(Y)))
TN = sum(numpy.logical_and(numpy.logical_not(Y_pred), numpy.logical_not(Y)))
FN = sum(numpy.logical_and(numpy.logical_not(Y_pred), Y))


BER = 1 - 0.5*(TP / (TP + FN) + TN / (TN + FP))


# In[280]:


answers['Q6'] = [TP, TN, FP, FN, BER]


# In[281]:


assertFloatList(answers['Q6'], 5)
print(answers['Q6'] )


# In[282]:


### Question 7


# In[283]:


mod = linear_model.LogisticRegression(C=1.0, class_weight = 'balanced')
mod.fit(X, Y)

Y_pred = mod.predict(X)

TP = sum(numpy.logical_and(Y_pred, Y))
FP = sum(numpy.logical_and(Y_pred, numpy.logical_not(Y)))
TN = sum(numpy.logical_and(numpy.logical_not(Y_pred), numpy.logical_not(Y)))
FN = sum(numpy.logical_and(numpy.logical_not(Y_pred), Y))


BER = 1 - 0.5*(TP / (TP + FN) + TN / (TN + FP))


# In[284]:


answers["Q7"] = [TP, TN, FP, FN, BER]
print(answers['Q7'])


# In[285]:


assertFloatList(answers['Q7'], 5)


# In[286]:


### Question 8


# In[287]:


scores = mod.decision_function(X_test)
scoreslabels = list(zip(scores, Y_test))

scoreslabels.sort(reverse = True)
sortedlabels = [s[1] for s in scoreslabels]
K = [1, 10, 100, 1000, 10000]
precisionList = []


for k in K:
    precisionList.append(sum(sortedlabels[:k]) / k)
    
    


# In[288]:


answers['Q8'] = precisionList
print(precisionList)


# In[289]:


assertFloatList(answers['Q8'], 5) #List of five floats


# In[290]:


f = open("answers_hw1.txt", 'w') # Write your answers to a file
f.write(str(answers) + '\n')
f.close()


# In[ ]:




