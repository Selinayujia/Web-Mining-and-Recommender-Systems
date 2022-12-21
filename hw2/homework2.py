#!/usr/bin/env python
# coding: utf-8

# In[546]:


import numpy
import urllib
import scipy.optimize
import random
from sklearn import linear_model
import gzip
from collections import defaultdict


# In[547]:


import warnings
warnings.filterwarnings("ignore")


# In[548]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[549]:


f = open("5year.arff", 'r')


# In[550]:


# Read and parse the data
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)


# In[551]:


X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]


# In[552]:


answers = {} # Your answers


# In[553]:


def accuracy(predictions, y):
    TP = sum(numpy.logical_and(predictions, y))
    FP = sum(numpy.logical_and(predictions, numpy.logical_not(y)))
    TN = sum(numpy.logical_and(numpy.logical_not(predictions), numpy.logical_not(y)))
    FN = sum(numpy.logical_and(numpy.logical_not(predictions), y))

    return (TP + TN) / (TP + FP + TN + FN)


# In[554]:


def BER(predictions, y):
    TP = sum(numpy.logical_and(predictions, y))
    FP = sum(numpy.logical_and(predictions, numpy.logical_not(y)))
    TN = sum(numpy.logical_and(numpy.logical_not(predictions), numpy.logical_not(y)))
    FN = sum(numpy.logical_and(numpy.logical_not(predictions), y))

    return 1 - 0.5*(TP / (TP + FN) + TN / (TN + FP))


# In[555]:


### Question 1


# In[556]:


mod = linear_model.LogisticRegression(C=1)
mod.fit(X,y)

pred = mod.predict(X)
acc1 = accuracy(pred, y)
ber1 = BER(pred, y)


# In[557]:


answers['Q1'] = [acc1, ber1] # Accuracy and balanced error rate


# In[558]:


assertFloatList(answers['Q1'], 2)


# In[559]:


### Question 2


# In[560]:


mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(X,y)

pred = mod.predict(X)
acc2 = accuracy(pred, y)
ber2 = BER(pred, y)


# In[561]:


answers['Q2'] = [acc2, ber2]


# In[562]:


assertFloatList(answers['Q2'], 2)


# In[563]:


### Question 3


# In[564]:


random.seed(3)
random.shuffle(dataset)


# In[565]:


X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]


# In[566]:


Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[567]:


len(Xtrain), len(Xvalid), len(Xtest), len(ytrain), len(yvalid), len(ytest)


# In[568]:


mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(Xtrain,ytrain)

predTrain = mod.predict(Xtrain)
predValid = mod.predict(Xvalid)
predTest = mod.predict(Xtest)

berTrain = BER(predTrain, ytrain)
berValid = BER(predValid, yvalid)
berTest = BER(predTest, ytest)


# In[569]:


answers['Q3'] = [berTrain, berValid, berTest]


# In[570]:


assertFloatList(answers['Q3'], 3)


# In[571]:


### Question 4


# In[572]:


c_s = [10**c for c in range(-4, 5)]
berList = []
for c in c_s:
    mod = linear_model.LogisticRegression(C=c, class_weight='balanced')

    mod.fit(Xvalid,yvalid)
    predValid = mod.predict(Xvalid)
    berValid = BER(predValid, yvalid)
    berList.append(berValid)


# In[573]:


answers['Q4'] = berList


# In[574]:


assertFloatList(answers['Q4'], 9)


# In[575]:


### Question 5


# In[576]:


c_ber_pair = zip(c_s, berList) 
bestC, ber5 = sorted(c_ber_pair, key=lambda x: x[-1])[0]


# In[577]:


answers['Q5'] = [bestC, ber5]


# In[578]:


assertFloatList(answers['Q5'], 2)


# In[579]:


### Question 6


# In[580]:


f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(eval(l))


# In[581]:


dataTrain = dataset[:9000]
dataTest = dataset[9000:]


# In[582]:


# Some data structures you might want

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {} # To retrieve a rating for a specific user/item pair
ratings = []
for d in dataTrain:
    user = d["user_id"]
    item = d["book_id"]
    rating = d["rating"]
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)
    ratingDict[f'{user}_{item}'] = rating
    ratings.append(rating)
    


# In[583]:


def Jaccard(s1, s2):
    result = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) != 0 else 0
    return result


# In[584]:


def mostSimilar(i, N):
    similarities = [(Jaccard(usersPerItem[i], usersPerItem[item]), item) for item in usersPerItem if item != i]
    return sorted(similarities, reverse=True, key=lambda x : x[0])[:N]

    


# In[585]:


answers['Q6'] = mostSimilar('2767052', 10)


# In[586]:


assert len(answers['Q6']) == 10
assertFloatList([x[0] for x in answers['Q6']], 10)
print(answers['Q6'])


# In[587]:


### Question 7


# In[588]:


userAverages = {}
itemAverages = {}

for u in itemsPerUser:
    rs = [ratingDict[f'{u}_{i}'] for i in itemsPerUser[u]]
    userAverages[u] = sum(rs) / len(rs)
    
for i in usersPerItem:
    rs = [ratingDict[f'{u}_{i}'] for u in usersPerItem[i]]
    itemAverages[i] = sum(rs) / len(rs)
    
ratingMean = sum(ratings)/len(ratings)


# In[589]:


def predict_rating(u, i):
    differences = []
    similarities = []                
    for j in itemsPerUser[u]:
        if j != i:
            r_u_j = ratingDict[f'{u}_{j}']
            average_j_rating = itemAverages[j] # rj bar
            difference = r_u_j - average_j_rating # r_u_j - rj bar
            similarity = Jaccard(usersPerItem[i], usersPerItem[j])
            differences.append(difference)
            similarities.append(similarity)
    
    weighted = zip(differences, similarities)
    result = itemAverages[i] + (sum([d*s for d, s in weighted]) / sum(similarities)) if sum(similarities) != 0 else ratingMean
    return result


# In[590]:


pred_rs = []
actual_rs = []
for d in dataTest:
    pred_rs.append(predict_rating(d['user_id'], d['book_id']))
    actual_rs.append(d['rating'])
    
    
def MSE(pred_rs, actual_rs):
    differences = [(pred_r - actual_r)**2 for pred_r, actual_r in zip(pred_rs,actual_rs)]
    return sum(differences) / len(differences)

mse7= MSE(pred_rs, actual_rs)


# In[591]:


answers['Q7'] = mse7


# In[592]:


assertFloat(answers['Q7'])
print(mse7)


# In[593]:


### Question 8


# In[594]:


def predict_rating_2(u, i):
    differences = []
    similarities = []                
    for v in usersPerItem[i]:
        if v != u:
            r_v_i = ratingDict[f'{v}_{i}']
            average_v_rating = userAverages[v] # rv bar
            difference = r_v_i - average_v_rating # r_v_i - rv bar
            similarity = Jaccard(itemsPerUser[u], itemsPerUser[v])
            differences.append(difference)
            similarities.append(similarity)
    
    weighted = zip(differences, similarities)
    result = itemAverages[i] + (sum([d*s for d, s in weighted]) / sum(similarities)) if sum(similarities) != 0 else ratingMean
    return result


# In[595]:


pred_rs = []
actual_rs = []
for d in dataTest:
    pred_rs.append(predict_rating_2(d['user_id'], d['book_id']))
    actual_rs.append(d['rating'])
    
    
def MSE(pred_rs, actual_rs):
    differences = [(pred_r - actual_r)**2 for pred_r, actual_r in zip(pred_rs,actual_rs)]
    return sum(differences) / len(differences)

mse8= MSE(pred_rs, actual_rs)


# In[596]:


answers['Q8'] = mse8


# In[597]:


assertFloat(answers['Q8'])


# In[598]:


f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()

