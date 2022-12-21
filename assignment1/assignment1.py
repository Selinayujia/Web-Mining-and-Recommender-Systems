#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[4]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[5]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[6]:


answers = {}


# In[7]:


# Some data structures that will be useful


# In[402]:


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)


# In[403]:


len(allRatings)


# In[451]:


ratingsTrain = allRatings[:199000]
ratingsValid = allRatings[199000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# In[452]:


##################################################
# Rating prediction (CSE258 only)                #
##################################################


# In[453]:


### Question 9


# In[454]:


def MSE(pred_rs, actual_rs):
    differences = [(pred_r - actual_r)**2 for pred_r, actual_r in zip(pred_rs,actual_rs)]
    return sum(differences) / len(differences)


# In[455]:


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


# In[461]:


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


# In[462]:


#validating
preds = []
valid_labels  = []

for u, b, r in ratingsValid:
    valid_labels.append(r)
    if u in userBiases and b in itemBiases:
        preds.append(prediction(u, b))
    else:
        user_ratings = [user_rating[1] for user_rating in ratingsPerUser[u]]
        if u in userBiases and b in itemBiases:
            preds.append(prediction(u, b))
        elif u not in userBiases and b not in itemBiases:
            preds.append(ratingMean)
        elif u not in userBiases:
            preds.append(ratingMean + itemBiases[b])
        else:
            preds.append(ratingMean + userBiases[u])
validMSE = MSE(preds, valid_labels)


# In[463]:


answers['Q9'] = validMSE
print(validMSE)


# In[464]:


### Question 11


# In[465]:


userBiases = defaultdict(float)
itemBiases = defaultdict(float)

lambs = [10**(i/10) for i in range(-52, -45)]
mses = []
for lamb in lambs:
    scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),
                                 derivative, args = (labels, lamb))

    preds = []
    valid_labels  = []

    for u, b, r in ratingsValid:
        valid_labels.append(r)
        if u in userBiases and b in itemBiases:
            preds.append(prediction(u, b))
        elif u not in userBiases and b not in itemBiases:
            preds.append(ratingMean)
        elif u not in userBiases:
            preds.append(ratingMean + itemBiases[b])
        else:
            preds.append(ratingMean + userBiases[u])
        
    mses.append(MSE(preds, valid_labels))

validMSE = min(mses)
lamb = lambs[mses.index(validMSE)]


# In[469]:


answers['Q11'] = (lamb, validMSE)
print(lamb), print(mses)


# In[470]:


print(lambs)


# In[471]:



ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in allRatings:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    
labels = [r[2] for r in allRatings]
ratingMean = sum(labels)/len(allRatings)

users = list(ratingsPerUser.keys())
items = list(ratingsPerItem.keys())
nUsers = len(users)
nItems = len(items)

alpha = ratingMean

userBiases = defaultdict(float)
itemBiases = defaultdict(float)


scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),
                             derivative, args = (labels, lamb))
predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
 
    if u in userBiases and b in itemBiases:
        predictions.write(u + ',' + b + ',' + str(prediction(u, b)) + "\n")
    elif u not in userBiases and b not in itemBiases:
        predictions.write(u + ',' + b + ',' + str(ratingMean) + "\n")
    elif u not in userBiases:
        
        predictions.write(u + ',' + b + ',' + str(ratingMean + itemBiases[b]) + "\n")
    else:
        predictions.write(u + ',' + b + ',' + str(ratingMean + userBiases[u]) + "\n")
    
predictions.close()


# In[196]:


##################################################
# Read prediction                                #
##################################################


# In[473]:


# Give a recommendation based on the popularity of the book
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

ratingsTrain = allRatings[:199500]
ratingsValid = allRatings[199500:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)


# In[474]:


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


# In[475]:


validWithNegative = sampleNegative(allRatings, ratingsValid)


# In[476]:


def recommendationWThreshold(totalRead, mostPopular, t):
    recommendation = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        recommendation.add(i)
        if count > int(totalRead*(t/100)): break
    return recommendation


# In[477]:


# Evaluating accuracy of recommendations with different thresholds
thresholds = [i/10 for i in range(700,720)]
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


# In[478]:


answers['Q2'] = [threshold, acc2]
print(threshold, acc2)


# In[479]:


def Jaccard(s1, s2):
    result = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) != 0 else 0
    return result


# In[480]:


ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    
thresholds = [i for i in range(40,51)] 
# Here threshold is defined as the percentage of the jaccard similarity
acc = []
for threshold in thresholds:
    correct = 0
    for u, b, r, read in validWithNegative:
        books_u_read = [book_rating[0] for book_rating in ratingsPerUser[u]]
        users_read_b = [user_rating[0] for user_rating in ratingsPerItem[b]]

        book_jaccard_scores = []
        for book in books_u_read:
            users_read_book = [user_rating[0] for user_rating in ratingsPerItem[book]]
            book_jaccard_scores.append(Jaccard(set(users_read_b), set(users_read_book)))

        user_jaccard_scores = []
        for user in users_read_b:
            books_user_read = [user_rating[0] for user_rating in ratingsPerUser[user]]
            user_jaccard_scores.append(Jaccard(set(books_u_read), set(books_user_read)))


        if not book_jaccard_scores and not user_jaccard_scores:
            jaccard = 0
        elif not user_jaccard_scores:
            jaccard = max(max(book_jaccard_scores), sum(book_jaccard_scores))*0.5
        elif not book_jaccard_scores:
            jaccard = max(max(user_jaccard_scores), sum(user_jaccard_scores))*0.5
        else:
            book_jaccard = max(max(book_jaccard_scores), sum(book_jaccard_scores))
            user_jaccard = max(max(user_jaccard_scores), sum(user_jaccard_scores))
            jaccard = 0.5*book_jaccard + 0.5*user_jaccard

            if jaccard >= threshold/1000: # this current book is considered as a good fit
                if read: correct +=1
            else:
                if not read: correct +=1
    acc.append(correct/len(validWithNegative))
    
threshold_new, acc3_5 = (thresholds[acc.index(max(acc))], max(acc))
print(threshold_new, acc3_5)


# In[481]:


thresholds = [i/100 for i in range(50,65)] 

# Here threshold is defined as the percentage of the jaccard similarity
acc = []
for threshold in thresholds:
    correct = 0
    for u, b, r, read in validWithNegative:
        books_u_read = [book_rating[0] for book_rating in ratingsPerUser[u]]
        users_read_b = [user_rating[0] for user_rating in ratingsPerItem[b]]

        book_jaccard_scores = []
        for book in books_u_read:
            users_read_book = [user_rating[0] for user_rating in ratingsPerItem[book]]
            book_jaccard_scores.append(Jaccard(set(users_read_b), set(users_read_book)))

        user_jaccard_scores = []
        for user in users_read_b:
            books_user_read = [user_rating[0] for user_rating in ratingsPerUser[user]]
            user_jaccard_scores.append(Jaccard(set(books_u_read), set(books_user_read)))


        if not book_jaccard_scores and not user_jaccard_scores:
            jaccard = 0
        elif not user_jaccard_scores:
            jaccard = max(max(book_jaccard_scores), sum(book_jaccard_scores))*threshold
        elif not book_jaccard_scores:
            jaccard = max(max(user_jaccard_scores), sum(user_jaccard_scores))*(1-threshold)
        else:
            book_jaccard = max(max(book_jaccard_scores), sum(book_jaccard_scores))
            user_jaccard = max(max(user_jaccard_scores), sum(user_jaccard_scores))
            jaccard = threshold*book_jaccard + (1-threshold)*user_jaccard

            if jaccard >= 48/1000: # this current book is considered as a good fit
                if read: correct +=1
            else:
                if not read: correct +=1
    acc.append(correct/len(validWithNegative))
    
threshold_new_1, acc3_25 = (thresholds[acc.index(max(acc))], max(acc))
print(threshold_new_1, acc3_25)


# In[501]:


# Evaluating accuracy of recommendations with the BEST jaccard threshold and popularity threshold
bestJaccardThreshold = 80
bestPopularityThreshold = 71.8 #71.8 #71.2
bestTheta = 0.63 #0.62 #0.55

recommendation = recommendationWThreshold(totalRead, mostPopular,bestPopularityThreshold)
# Here threshold is defined as the percentage of the jaccard similarity
correct = 0
for u, b, r, read in validWithNegative:
    books_u_read = [book_rating[0] for book_rating in ratingsPerUser[u]]
    users_read_b = [user_rating[0] for user_rating in ratingsPerItem[b]]
    
    book_jaccard_scores = []
    for book in books_u_read:
        users_read_book = [user_rating[0] for user_rating in ratingsPerItem[book]]
        book_jaccard_scores.append(Jaccard(set(users_read_b), set(users_read_book)))
        
    user_jaccard_scores = []
    for user in users_read_b:
        books_user_read = [user_rating[0] for user_rating in ratingsPerUser[user]]
        user_jaccard_scores.append(Jaccard(set(books_u_read), set(books_user_read)))
    
            
    if not book_jaccard_scores and not user_jaccard_scores:
        jaccard = 0
    elif not user_jaccard_scores:
        jaccard = max(max(book_jaccard_scores), sum(book_jaccard_scores))*bestTheta
    elif not book_jaccard_scores:
        jaccard = max(max(user_jaccard_scores), sum(user_jaccard_scores))*(1-bestTheta)
    else:
        book_jaccard = max(max(book_jaccard_scores), sum(book_jaccard_scores))
        user_jaccard = max(max(user_jaccard_scores), sum(user_jaccard_scores))
        jaccard = (bestTheta)*book_jaccard + (1-bestTheta)*user_jaccard

    if b in recommendation:
        if read: correct +=1
    else:
        if jaccard >= bestJaccardThreshold/1000:
            if read: correct +=1
        else:
            if not read: correct += 1
acc4 = correct/len(validWithNegative)
    
print(acc4) # 0.7562


# In[502]:


ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in allRatings:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# In[503]:


bestJaccardThreshold = 80#80
bestPopularityThreshold = 71.8 # 71.2
bestTheta = 0.63 #0.6
recommendation = recommendationWThreshold(totalRead, mostPopular,bestPopularityThreshold)
predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    
    books_u_read = [book_rating[0] for book_rating in ratingsPerUser[u]]
    users_read_b = [user_rating[0] for user_rating in ratingsPerItem[b]]
    
    book_jaccard_scores = []
    for book in books_u_read:
        users_read_book = [user_rating[0] for user_rating in ratingsPerItem[book]]
        book_jaccard_scores.append(Jaccard(set(users_read_b), set(users_read_book)))
        
    user_jaccard_scores = []
    for user in users_read_b:
        books_user_read = [user_rating[0] for user_rating in ratingsPerUser[user]]
        user_jaccard_scores.append(Jaccard(set(books_u_read), set(books_user_read)))
    
            
    if not book_jaccard_scores and not user_jaccard_scores:
        jaccard = 0
    elif not user_jaccard_scores:
        jaccard = max(max(book_jaccard_scores), sum(book_jaccard_scores))*bestTheta
    elif not book_jaccard_scores:
        jaccard = max(max(user_jaccard_scores), sum(user_jaccard_scores))*(1-bestTheta)
    else:
        book_jaccard = max(max(book_jaccard_scores), sum(book_jaccard_scores))
        user_jaccard = max(max(user_jaccard_scores), sum(user_jaccard_scores))
        jaccard = (bestTheta)*book_jaccard + (1-bestTheta)*user_jaccard
        
    if b in recommendation:
        predictions.write(u + ',' + b + ",1\n")
    else:
        if jaccard >= bestJaccardThreshold/1000:
            predictions.write(u + ',' + b + ",1\n")
        else:
            predictions.write(u + ',' + b + ",0\n")
predictions.close()


# In[ ]:




