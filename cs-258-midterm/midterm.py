#!/usr/bin/env python
# coding: utf-8

# In[29]:


import json
import gzip
import math
from collections import defaultdict
import numpy
from sklearn import linear_model


# In[30]:


# This will suppress any warnings, comment out if you'd like to preserve them
import warnings
warnings.filterwarnings("ignore")


# In[31]:


# Check formatting of submissions
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[32]:


answers = {}


# In[33]:


f = open("spoilers.json.gz", 'r')


# In[34]:


dataset = []
for l in f:
    d = eval(l)
    dataset.append(d)


# In[35]:


f.close()


# In[36]:


# A few utility data structures
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    u,i = d['user_id'],d['book_id']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)

# Sort reviews per user by timestamp
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['timestamp'])
    
# Same for reviews per item
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['timestamp'])


# In[37]:


# E.g. reviews for this user are sorted from earliest to most recent
[d['timestamp'] for d in reviewsPerUser['b0d7e561ca59e313b728dc30a5b1862e']]


# In[38]:


### 1a


# In[39]:


def MSE(y, y_pred):
    return numpy.square(numpy.subtract(y,y_pred)).mean()


# In[40]:


y = []
ypred = []
for u in reviewsPerUser:
    user_data = reviewsPerUser[u]
    if len(user_data) > 1:
        y.append(user_data[-1]['rating'])
        ypred.append(sum([user_data[i]['rating'] for i in range(len(user_data) -1)])/(len(user_data) -1))


# In[41]:


answers['Q1a'] = MSE(y,ypred)


# In[42]:


assertFloat(answers['Q1a'])


# In[43]:


### 1b


# In[44]:


y = []
ypred = []
for i in reviewsPerItem:
    item_data = reviewsPerItem[i]
    if len(item_data) > 1:
        y.append(item_data[-1]['rating'])
        ypred.append(sum([item_data[ind]['rating'] for ind in range(len(item_data) -1)])/(len(item_data) -1))


# In[45]:


answers['Q1b'] = MSE(y,ypred)


# In[46]:


assertFloat(answers['Q1b'])


# In[47]:


### 2


# In[48]:


answers['Q2'] = []
for N in [1,2,3]:
    y = []
    ypred = []
    for u in reviewsPerUser:
        user_data = reviewsPerUser[u]
        if len(user_data) > 1:
            y.append(user_data[-1]['rating'])
            start = len(user_data) - 1 - N if len(user_data) > (1 + N) else 0
            end = len(user_data) - 1
            prediction = sum([user_data[i]['rating'] for i in range(start, end)])/(end-start)
            ypred.append(prediction)
    answers['Q2'].append(MSE(y,ypred))


# In[49]:


assertFloatList(answers['Q2'], 3)


# In[50]:


### 3a


# In[54]:


def feature3(N, u): # For a user u and a window size of N
    user_data = reviewsPerUser[u]
    if len(user_data) > (1 + N):
        start = len(user_data) - 1 - N
    else:
        return []
    end = len(user_data) - 1
    feature_v = [user_data[i]['rating'] for i in range(start, end)] # theta n - theta 1
    feature_v.append(1) # theta 0
    feature_v.reverse()
    return feature_v


# In[55]:


answers['Q3a'] = [feature3(2,dataset[0]['user_id']), feature3(3,dataset[0]['user_id'])]


# In[56]:


assert len(answers['Q3a']) == 2
assert len(answers['Q3a'][0]) == 3
assert len(answers['Q3a'][1]) == 4


# In[57]:


### 3b


# In[58]:


answers['Q3b'] = []

for N in [1,2,3]:
    x = []
    y = []
    ypred = []
    for u in reviewsPerUser:
        f = feature3(N, u)
        if len(f) > 0:
            y.append(reviewsPerUser[u][-1]['rating'])
            x.append(f)
    theta,residuals,rank,s = numpy.linalg.lstsq(x, y, rcond=None)
    ypred = numpy.dot(x, theta)
    mse = MSE(y,ypred)
    answers['Q3b'].append(mse)


# In[59]:


assertFloatList(answers['Q3b'], 3)


# In[60]:


### 4a


# In[61]:


globalAverage = [d['rating'] for d in dataset]
globalAverage = sum(globalAverage) / len(globalAverage)


# In[62]:


def featureMeanValue(N, u): # For a user u and a window size of N
    user_data = reviewsPerUser[u]
    if len(user_data) > 1:
        feature_v = [user_data[i]['rating'] for i in range(len(user_data) - 1)] # theta n-x - theta 1
        feature_v.reverse()  
        avg = sum(feature_v)/len(feature_v)  
        if N <= len(feature_v):
            return [1] + feature_v[:N]
        else:
            return [1] + [feature_v[i] if i < len(feature_v) else avg for i in range(N)]
    else:
        return [1] + [globalAverage for i in range(N)]
    


# In[63]:


def featureMissingValue(N, u):
    user_data = reviewsPerUser[u]
    if len(user_data) > 1:
        feature_v = [[0, user_data[i]['rating']] for i in range(len(user_data) - 1)] # theta n-x - theta 1
        feature_v.reverse()  
        if N <= len(feature_v):
            lst = feature_v[:N]
        else:
            lst = [feature_v[i] if i < len(feature_v) else [1,0] for i in range(N)]
    else:
        lst = [[1,0] for i in range(N)]
    return [1] + [item for sublist in lst for item in sublist]


# In[64]:


answers['Q4a'] = [featureMeanValue(10, dataset[0]['user_id']), featureMissingValue(10, dataset[0]['user_id'])]


# In[65]:


assert len(answers['Q4a']) == 2
assert len(answers['Q4a'][0]) == 11
assert len(answers['Q4a'][1]) == 21


# In[66]:


### 4b


# In[67]:


answers['Q4b'] = []

for featFunc in [featureMeanValue, featureMissingValue]:
    N = 10
    x = []
    y = []
    ypred = []
    for u in reviewsPerUser:
        f = featFunc(N, u)
        y.append(reviewsPerUser[u][-1]['rating'])
        x.append(f)
    theta,residuals,rank,s = numpy.linalg.lstsq(x, y, rcond=None)
    ypred = numpy.dot(x, theta)
    mse = MSE(y,ypred)
    answers['Q4b'].append(mse)


# In[68]:


assertFloatList(answers["Q4b"], 2)


# In[69]:


### 5


# In[70]:


def feature5(sentence):
    length = len(sentence)
    ex_num = sentence.count('!')
    cap_num = sum([1 if c.isupper() else 0 for string in sentence for c in string])
    return [1, length, ex_num, cap_num]


# In[71]:


y = []
X = []
for d in dataset:
    for spoiler,sentence in d['review_sentences']:
        X.append(feature5(sentence))
        y.append(spoiler)


# In[72]:


answers['Q5a'] = X[0]


# In[73]:


mod = linear_model.LogisticRegression(C=1.0, class_weight = 'balanced')
mod.fit(X, y)
ypred = mod.predict(X)
TP = sum(numpy.logical_and(ypred, y))
FP = sum(numpy.logical_and(ypred, numpy.logical_not(y)))
TN = sum(numpy.logical_and(numpy.logical_not(ypred), numpy.logical_not(y)))
FN = sum(numpy.logical_and(numpy.logical_not(ypred), y))


BER = 1 - 0.5*(TP / (TP + FN) + TN / (TN + FP))


# In[74]:


answers['Q5b'] = [TP, TN, FP, FN, BER]


# In[75]:


assert len(answers['Q5a']) == 4
assertFloatList(answers['Q5b'], 5)


# In[76]:


### 6


# In[77]:



def feature6(review):
    sentences = d['review_sentences']
    features = [sentences[i][0] for i in range(0,5)]
    prev_feature = feature5(sentences[5][1])
    return features + prev_feature
    
    


# In[78]:


y = []
X = []
for d in dataset:
    sentences = d['review_sentences']
    if len(sentences) < 6: continue
    X.append(feature6(d))
    y.append(sentences[5][0])

mod = linear_model.LogisticRegression(C=1.0, class_weight = 'balanced')
mod.fit(X, y)
ypred = mod.predict(X)
TP = sum(numpy.logical_and(ypred, y))
FP = sum(numpy.logical_and(ypred, numpy.logical_not(y)))
TN = sum(numpy.logical_and(numpy.logical_not(ypred), numpy.logical_not(y)))
FN = sum(numpy.logical_and(numpy.logical_not(ypred), y))


BER = 1 - 0.5*(TP / (TP + FN) + TN / (TN + FP))


# In[79]:


answers['Q6a'] = X[0]


# In[80]:


answers['Q6b'] = BER


# In[81]:


assert len(answers['Q6a']) == 9
assertFloat(answers['Q6b'])


# In[82]:


### 7


# In[83]:


# 50/25/25% train/valid/test split
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[84]:


bers = []
for c in [0.01, 0.1, 1, 10, 100]:
    mod = linear_model.LogisticRegression(C=c, class_weight = 'balanced')
    mod.fit(Xtrain, ytrain)
    ypred = mod.predict(Xvalid)
    TP = sum(numpy.logical_and(ypred, yvalid))
    FP = sum(numpy.logical_and(ypred, numpy.logical_not(yvalid)))
    TN = sum(numpy.logical_and(numpy.logical_not(ypred), numpy.logical_not(yvalid)))
    FN = sum(numpy.logical_and(numpy.logical_not(ypred), yvalid))
    BER = 1 - 0.5*(TP / (TP + FN) + TN / (TN + FP))
    bers.append(BER)
    
bestC = [0.01, 0.1, 1, 10, 100][bers.index(min(bers))]

mod = linear_model.LogisticRegression(C=bestC, class_weight = 'balanced')
mod.fit(Xtrain, ytrain)

ypred = mod.predict(Xtest)
TP = sum(numpy.logical_and(ypred, ytest))
FP = sum(numpy.logical_and(ypred, numpy.logical_not(ytest)))
TN = sum(numpy.logical_and(numpy.logical_not(ypred), numpy.logical_not(ytest)))
FN = sum(numpy.logical_and(numpy.logical_not(ypred), ytest))
ber = 1 - 0.5*(TP / (TP + FN) + TN / (TN + FP))


# In[85]:


answers['Q7'] = bers + [bestC] + [ber]


# In[86]:


assertFloatList(answers['Q7'], 7)


# In[87]:


### 8


# In[88]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[89]:


# 75/25% train/test split
dataTrain = dataset[:15000]
dataTest = dataset[15000:]


# In[90]:


# A few utilities

itemAverages = defaultdict(list)
ratingMean = []

for d in dataTrain:
    itemAverages[d['book_id']].append(d['rating'])
    ratingMean.append(d['rating'])

for i in itemAverages:
    itemAverages[i] = sum(itemAverages[i]) / len(itemAverages[i])

ratingMean = sum(ratingMean) / len(ratingMean)


# In[91]:


reviewsPerUser = defaultdict(list)
usersPerItem = defaultdict(set)

for d in dataTrain:
    u,i = d['user_id'], d['book_id']
    reviewsPerUser[u].append(d)
    usersPerItem[i].add(u)


# In[92]:


# From my HW2 solution, welcome to reuse
def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        if item in itemAverages:
            return itemAverages[item]
        else:
            return ratingMean


# In[93]:


predictions = []
labels = []
for d in dataTest:
    predictions.append(predictRating(d['user_id'], d['book_id']))
    labels.append(d['rating'])
    
    
def MSE(pred_rs, actual_rs):
    differences = [(pred_r - actual_r)**2 for pred_r, actual_r in zip(pred_rs,actual_rs)]
    return sum(differences) / len(differences)


# In[94]:


answers["Q8"] = MSE(predictions, labels)


# In[95]:


assertFloat(answers["Q8"])


# In[96]:


### 9


# In[98]:


item_0_pred = []
item_0_label = []

item_1to5_pred = []
item_1to5_label = []

item_5_pred = []
item_5_label = []


for d in dataTest:
    item_num = len(usersPerItem[d['book_id']])
    prediction = predictRating(d['user_id'], d['book_id'])
    if item_num == 0:
        item_0_pred.append(prediction)
        item_0_label.append(d['rating'])
    elif item_num <= 5:
        item_1to5_pred.append(prediction)
        item_1to5_label.append(d['rating'])
    else:
        item_5_pred.append(prediction)
        item_5_label.append(d['rating'])


# In[99]:


mse0 = MSE(item_0_pred, item_0_label)
mse1to5 = MSE(item_1to5_pred, item_1to5_label)
mse5 = MSE(item_5_pred, item_5_label)


# In[100]:


answers["Q9"] = [mse0, mse1to5, mse5]


# In[101]:


assertFloatList(answers["Q9"], 3)


# In[102]:


### 10


# In[103]:


item_0_pred = []
item_0_label = []
itemsPerUser = defaultdict(set)
user_item_rating = {}
for d in dataTrain:
    u,i = d['user_id'], d['book_id']
    user_item_rating[(u,i)] = d['rating']
    itemsPerUser[u].add(i)

def predictRatingOptimizing(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        if item in itemAverages:
            return itemAverages[item]
        else:
            # items never seen before
            all_other_items = itemsPerUser[user]
            if len(all_other_items) == 0:
                return ratingMean
            else:
                all_other_item_rating = [user_item_rating[(user, item)] for item in all_other_items]
                return sum(all_other_item_rating) / len(all_other_item_rating)
            
for d in dataTest:
    item_num = len(usersPerItem[d['book_id']])
    prediction = predictRatingOptimizing(d['user_id'], d['book_id'])
    if item_num == 0:
        item_0_pred.append(prediction)
        item_0_label.append(d['rating'])
itsMSE = MSE(item_0_pred, item_0_label)
    


# In[104]:


answers["Q10"] = ("While in the previous approach when an item was never seen before, we return the average of all item all user's rating. However, the user who buys this cold star item may have a rating tendency (tend to overall give high ratings or overall give low rating) The approach here is that instead of taking average of all user's rating on all item, if the user is seen before in the training set, we take the average of all this user's previous rating on previous items they bought. If the user also didn't buy other things before, we then still return the old rating average among all user all item", itsMSE)


# In[105]:


assert type(answers["Q10"][0]) == str
assertFloat(answers["Q10"][1])


# In[106]:


f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




