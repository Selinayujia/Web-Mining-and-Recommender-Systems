#!/usr/bin/env python
# coding: utf-8

# In[258]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model


# In[259]:


import warnings
warnings.filterwarnings("ignore")


# In[260]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[261]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[262]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[263]:


answers = {}


# In[264]:


# Some data structures that will be useful


# In[265]:


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)


# In[266]:


len(allRatings)


# In[267]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# In[268]:


##################################################
# Rating prediction (CSE258 only)                #
##################################################


# In[269]:


### Question 9


# In[313]:


def MSE(pred_rs, actual_rs):
    differences = [(pred_r - actual_r)**2 for pred_r, actual_r in zip(pred_rs,actual_rs)]
    return sum(differences) / len(differences)


# In[341]:


def prediction(user, item):
    return alpha + userBiases[user] + itemBiases[item]

def unpack(theta):
    global alpha
    global userBiases
    global itemBiases
    alpha = theta[0]
    userBiases = dict(zip(users, theta[1:nUsers+1]))
    itemBiases = dict(zip(items, theta[1+nUsers:]))
    
def cost(theta, labels, lamb):
    unpack(theta)
    predictions = [prediction(u, b) for u, b, r in ratingsTrain]
    cost = MSE(predictions, labels)
    for u in userBiases:
        cost += lamb*userBiases[u]**2
    for i in itemBiases:
        cost += lamb*itemBiases[i]**2
    return cost

def derivative(theta, labels, lamb):
    unpack(theta)
    N = len(ratingsTrain)
    dalpha = 0
    dUserBiases = defaultdict(float)
    dItemBiases = defaultdict(float)
    for u, b, r in ratingsTrain:
        u,i = u, b
        pred = prediction(u, i)
        diff = pred - r
        dalpha += 2/N*diff
        dUserBiases[u] += 2/N*diff
        dItemBiases[i] += 2/N*diff
    for u in userBiases:
        dUserBiases[u] += 2*lamb*userBiases[u]
    for i in itemBiases:
        dItemBiases[i] += 2*lamb*itemBiases[i]
    dtheta = [dalpha] + [dUserBiases[u] for u in users] + [dItemBiases[i] for i in items]
    return np.array(dtheta)


# In[342]:


# Train
lamb = 1

labels = [r[2] for r in ratingsTrain]
ratingMean = sum(labels)/len(ratingsTrain)

users = list(ratingsPerUser.keys())
items = list(ratingsPerItem.keys())
nUsers = len(users)
nItems = len(items)

alpha = ratingMean

userBiases = defaultdict(float)
itemBiases = defaultdict(float)


scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),
                             derivative, args = (labels, lamb))


# In[343]:


#validating
preds = []
valid_labels  = []

for u, b, r in ratingsValid:
    valid_labels.append(r)
    if u in userBiases and b in itemBiases:
        preds.append(prediction(u, b))
    else:
        preds.append(ratingMean)
validMSE = MSE(preds, valid_labels)


# In[345]:


answers['Q9'] = validMSE
print(validMSE)


# In[346]:


assertFloat(answers['Q9'])


# In[347]:


### Question 10


# In[364]:


maxBeta = float(max(userBiases.values()))
minBeta = float(min(userBiases.values()))

maxUser, minUser = '', ''
for key in userBiases:
    if userBiases[key] == maxBeta:
        maxUser= key
    elif userBiases[key] == minBeta:
        minUser = key


# In[365]:


answers['Q10'] = [maxUser, minUser, maxBeta, minBeta]


# In[366]:


assert [type(x) for x in answers['Q10']] == [str, str, float, float]


# In[67]:


### Question 11


# In[ ]:


userBiases = defaultdict(float)
itemBiases = defaultdict(float)

lambs = [10**i for i in range(-6, 2)]
mses = []
for lamb in lambs:
    scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),
                                 derivative, args = (labels, lamb))
    print(lamb)

    preds = []
    valid_labels  = []

    for u, b, r in ratingsValid:
        valid_labels.append(r)
        if u in userBiases and b in itemBiases:
            preds.append(prediction(u, b))
        else:
            preds.append(ratingMean)
    mses.append(MSE(preds, valid_labels))

validMSE = min(mses)
lamb = lambs[mses.index(validMSE)]


# In[370]:


answers['Q11'] = (lamb, validMSE)


# In[371]:


assertFloat(answers['Q11'][0])
assertFloat(answers['Q11'][1])


# In[372]:


predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    # (etc.)
    
predictions.close()


# In[196]:


##################################################
# Read prediction                                #
##################################################


# In[280]:


### Question 1


# In[281]:


# Give a recommendation based on the popularity of the book
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

def recommendationOnPopularity(totalRead, mostPopular):
    recommendation = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        recommendation.add(i)
        if count > totalRead/2: break
    return recommendation


# In[282]:


def sampleNegative(allRatings, valid):
    validWithNeg = []
    userBookList = defaultdict(set)
    allBookList = set()
    for u, b, r in allRatings:
        userBookList[u].add(b)
        allBookList.add(b)
    
    for u, b, r in valid:
        validWithNeg.append((u,b,r,1)) # 1 stands for read
        aBookUserNeverRead = random.sample(allBookList - userBookList[u], 1)[0]
        validWithNeg.append((u, aBookUserNeverRead, 0, 0))
    return validWithNeg


# In[283]:


validWithNegative = sampleNegative(allRatings, ratingsValid)


# In[284]:


# Evaluating accuracy of recommendation
recommendation = recommendationOnPopularity(totalRead, mostPopular)
correct = 0
for u, b, r, read in validWithNegative:
    if b in recommendation:
        if read: correct +=1
    else:
        if not read: correct+=1
acc1 = correct/len(validWithNegative)


# In[285]:


answers['Q1'] = acc1
print(acc1)


# In[286]:


assertFloat(answers['Q1'])


# In[287]:


### Question 2


# In[288]:


def recommendationWThreshold(totalRead, mostPopular, t):
    recommendation = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        recommendation.add(i)
        if count > int(totalRead*(t/100)): break
    return recommendation


# In[289]:


# Evaluating accuracy of recommendations with different thresholds
thresholds = [i*5 for i in range(1,20)]
# Here threshold is defined as the percentage of the popularity book we are recommending (Q1 is 50)
acc = []
for threshold in thresholds:
    recommendation = recommendationWThreshold(totalRead, mostPopular, threshold)
    correct = 0
    for u, b, r, read in validWithNegative:
        if b in recommendation:
            if read: correct +=1
        else:
            if not read: correct+=1
    acc.append(correct/len(validWithNegative))
    
threshold, acc2 = (thresholds[acc.index(max(acc))], max(acc))


# In[290]:


answers['Q2'] = [threshold, acc2]
print(threshold, acc2)


# In[291]:


assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])


# In[292]:


### Question 3/4


# In[293]:


def Jaccard(s1, s2):
    result = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) != 0 else 0
    return result


# In[294]:


# Evaluating accuracy of recommendations with different jaccard threshold
thresholds = [i for i in range(1,51)] # 0.001 - 0.05

# Here threshold is defined as the percentage of the jaccard similarity
acc = []
for threshold in thresholds:
    correct = 0
    for u, b, r, read in validWithNegative:
        jaccard_scores = []
        books_u_read = [book_rating[0] for book_rating in ratingsPerUser[u]]
        users_read_b = [user_rating[0] for user_rating in ratingsPerItem[b]]
        for book in books_u_read:
            users_read_book = [user_rating[0] for user_rating in ratingsPerItem[book]]
            jaccard_scores.append(Jaccard(set(users_read_b), set(users_read_book)))
            
        if not jaccard_scores:
            jaccard = 0
        else:
            jaccard = max(max(jaccard_scores), sum(jaccard_scores)/len(jaccard_scores))
            
        if jaccard >= threshold/1000: # this current book is considered as a good fit
            if read: correct +=1
        else:
            if not read: correct +=1
    acc.append(correct/len(validWithNegative))
    
threshold, acc3 = (thresholds[acc.index(max(acc))], max(acc))


# In[295]:


# Evaluating accuracy of recommendations with the BEST jaccard threshold and popularity threshold

bestJaccardThreshold = 3/1000
bestPopularityThreshold = 75

# Here threshold is defined as the percentage of the jaccard similarity
correct = 0
for u, b, r, read in validWithNegative:
    jaccard_scores = []
    books_u_read = [book_rating[0] for book_rating in ratingsPerUser[u]]
    users_read_b = [user_rating[0] for user_rating in ratingsPerItem[b]]
    for book in books_u_read:
        users_read_book = [user_rating[0] for user_rating in ratingsPerItem[book]]
        jaccard_scores.append(Jaccard(set(users_read_b), set(users_read_book)))
            
    if not jaccard_scores:
        jaccard = 0
    else:
        jaccard = max(max(jaccard_scores), sum(jaccard_scores)/len(jaccard_scores))
        
    if b in recommendation and jaccard >= bestJaccardThreshold:
        if read: correct +=1
    else:
        if not read: correct += 1
acc4 = correct/len(validWithNegative)
    
print(acc4)


# In[296]:


answers['Q3'] = acc3
answers['Q4'] = acc4


# In[297]:


assertFloat(answers['Q3'])
assertFloat(answers['Q4'])


# In[298]:


predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # (etc.)

predictions.close()


# In[299]:


answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"


# In[300]:


assert type(answers['Q5']) == str


# In[373]:


f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




