#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
import math
import numpy
import random
import sklearn
import string
from collections import defaultdict
from gensim.models import Word2Vec
from nltk.stem.porter import *
from sklearn import linear_model
from sklearn.manifold import TSNE
import dateutil


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


dataset = []

f = gzip.open("young_adult_20000.json.gz")
for l in f:
    d = eval(l)
    dataset.append(d)
    if len(dataset) >= 20000:
        break
        
f.close()


# In[ ]:


d_train = dataset[:10000]
d_test = dataset[10000:]


# In[52]:


answers = {}


# In[ ]:


### Question 1


# In[19]:


unigram_wordCount = defaultdict(int)
bigram_wordCount = defaultdict(int)
uni_bi_wordCount = defaultdict(int)

punctuation = set(string.punctuation)
for d in d_train:
    r = ''.join([c for c in d['review_text'].lower()                  if not c in punctuation])
    uni_w = r.split()
    bi_w = [' '.join(pair) for pair in list(zip(uni_w[:-1],uni_w[1:]))]
    for i in range(len(bi_w)):
        unigram_wordCount[uni_w[i]] += 1
        bigram_wordCount[bi_w[i]] += 1
        uni_bi_wordCount[uni_w[i]] += 1
        uni_bi_wordCount[bi_w[i]] += 1
        
    if(len(uni_w)):
        unigram_wordCount[uni_w[-1]] += 1
        uni_bi_wordCount[uni_w[-1]] += 1
len(unigram_wordCount),len(bigram_wordCount),len(uni_bi_wordCount)


# In[47]:


def get_common_ngrams(dic, cutoff=1000):
    counts = [(dic[ngram], ngram) for ngram in dic]
    counts.sort()
    counts.reverse()
    words = [x[1] for x in counts[:cutoff]]
    wordId = dict(zip(words, range(len(words))))
    return words, wordId

ngrams_dic = {'mostCommonUnigrams': get_common_ngrams(unigram_wordCount),               'mostCommonBigrams': get_common_ngrams(bigram_wordCount),               'mostCommonBoth':get_common_ngrams(uni_bi_wordCount)}


# In[48]:


def feature(datum, model):
    words, wordId = ngrams_dic[model]
    
    feat = [0]*len(words)
    
    r = ''.join([c for c in datum['review_text'].lower() if not c in punctuation])
    if model == 'mostCommonUnigrams':
        r_ngram = r.split()
    elif model == 'mostCommonBigrams':
        r_ngram = [' '.join(pair) for pair in list(zip(r.split()[:-1],r.split()[1:]))]
    else:
        r_ngram = r.split() + [' '.join(pair) for pair in list(zip(r.split()[:-1],r.split()[1:]))]
    
    for w in r_ngram:
        if w in words:
            feat[wordId[w]] += 1
    feat.append(1) 
    return feat


# In[49]:



def MSE(pred, y):
    differences = [(pred - y)**2 for pred, y in zip(pred, y)]
    return sum(differences) / len(differences)
    


# In[58]:


for q, model in ('Q1a', 'mostCommonUnigrams'), ('Q1b', 'mostCommonBigrams'), ('Q1c', 'mostCommonBoth'):
    X_train = [feature(d, model) for d in d_train]
    y_train = [d['rating'] for d in d_train]
    
    words, wordId = ngrams_dic[model]

    # Regularized regression
    clf = linear_model.Ridge(1.0, fit_intercept=False) 
    clf.fit(X_train, y_train)
    theta = clf.coef_
    
    wordSort = list(zip(theta[:-1], words))
    wordSort.sort()
    
    X_test = [feature(d, model) for d in d_test]
    y_test = [d['rating'] for d in d_test]
    predictions = clf.predict(X_test)
    
    mse = MSE(predictions, y_test)
    most_negative = [x[1] for x in wordSort[:5]]
    most_positive = [x[1] for x in wordSort[-5:]]
    
    print(f'{model}, {mse},{most_negative}, {most_positive}')
    answers[q] = [float(mse), most_negative, most_positive]
    


# In[59]:


for q in 'Q1a', 'Q1b', 'Q1c':
    assert len(answers[q]) == 3
    assertFloat(answers[q][0])
    assert [type(x) for x in answers[q][1]] == [str]*5
    assert [type(x) for x in answers[q][2]] == [str]*5


# In[ ]:


### Question 2


# In[62]:


def Cosine(x1,x2):
    numer = 0
    norm1 = 0
    norm2 = 0
    for a1,a2 in zip(x1,x2):
        numer += a1*a2
        norm1 += a1**2
        norm2 += a2**2
    if norm1*norm2:
        return numer / math.sqrt(norm1*norm2)
    return 0


# In[63]:


words, _ = ngrams_dic['mostCommonUnigrams']
# df
df = defaultdict(int)
for d in d_train:
    r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
    for w in set(r.split()):
        df[w] += 1 # a word only count once in a review


# In[64]:


# tf for review 1 in train dataset
tf = defaultdict(int)
r = ''.join([c for c in d_train[0]['review_text'].lower() if not c in punctuation])
for w in r.split():
    tf[w] = 1
    
tfidfQuery = [tf[w] * math.log2(len(dataset) / df[w]) for w in words]


# In[67]:


similarities = []
# tf for rest of the review in train dataset
for rev in d_train[1:]:
    tf = defaultdict(int)
    r = ''.join([c for c in rev['review_text'].lower() if not c in punctuation])
    for w in r.split():
        tf[w] = 1
    tfidf2 = [tf[w] * math.log2(len(dataset) / df[w]) for w in words]
    similarities.append((Cosine(tfidfQuery, tfidf2), rev['review_text']))
    
similarities.sort(reverse=True)
sim, review = similarities[0]


# In[68]:


answers['Q2'] = [sim, review]


# In[69]:


assert len(answers['Q2']) == 2
assertFloat(answers['Q2'][0])
assert type(answers['Q2'][1]) == str


# In[ ]:


### Question 3


# In[71]:


dataset[0]


# In[72]:


reviewsPerUser = defaultdict(list)
book_to_review = defaultdict(list)


# In[73]:


for d in dataset:
    reviewDicts.append(d)
    book_to_review[d['book_id']].append(d['review_text'])
    reviewsPerUser[d['user_id']].append((dateutil.parser.parse(d['date_added']), d['book_id']))


# In[77]:


#word2vec (gensim)
#Tokenize the reviews, so that each review becomes a list of words

reviewLists = []
for u in reviewsPerUser:
    rl = list(reviewsPerUser[u])
    rl.sort()
    reviewLists.append([x[1] for x in rl])

model10 = Word2Vec(reviewLists,
                 min_count=1, # Words/items with fewer instances are discarded
                 vector_size=10, # Model dimensionality
                 window=3, # Window size
                 sg=1) # Skip-gram model
similarities = []
for b in model10.wv.similar_by_word(dataset[0]['book_id'])[:5]:
    similarities.append(b)
    print(book_to_review[b[0]])
    


# In[79]:


answers['Q3'] = similarities


# In[80]:


assert len(answers['Q3']) == 5
assert [type(x[0]) for x in answers['Q3']] == [str]*5
assertFloatList([x[1] for x in answers['Q3']], 5)


# In[ ]:


### Question 4


# In[82]:


ratingMean = sum([d['rating'] for d in dataset]) / len(dataset)
itemAverages = defaultdict(list)
reviewsPerUser = defaultdict(list)
    
for d in dataset:
    i = d['book_id']
    u = d['user_id']
    itemAverages[i].append(d['rating'])
    reviewsPerUser[u].append(d)
    
for i in itemAverages:
    itemAverages[i] = sum(itemAverages[i]) / len(itemAverages[i])
    
def predictRating(user,item):
    ratings = []
    similarities = []
    if not str(item) in model10.wv:
        return ratingMean
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        if str(i2) in model10.wv:
            similarities.append(Cosine(model10.wv[item], model10.wv[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + (sum(weightedRatings) / sum(similarities))
    else:
        return ratingMean

y = []
pred_y = []
for d in dataset[:1000]:
    y.append(d['rating'])
    pred_y.append(predictRating(d['user_id'],d['book_id']))
mse4 = MSE(pred_y, y)


# In[83]:


answers['Q4'] = mse4


# In[84]:


assertFloat(answers['Q4'])


# In[ ]:


### Q5


# In[87]:


model10 = Word2Vec(reviewLists,
                 min_count=5, # Words/items with fewer instances are discarded
                 vector_size=10, # Model dimensionality
                 window=3, # Window size
                 sg=1) # Skip-gram model
y = []
pred_y = []
for d in dataset[:1000]:
    y.append(d['rating'])
    pred_y.append(predictRating(d['user_id'],d['book_id']))
mse5 = MSE(pred_y, y)
print(mse5)


# In[88]:


answers['Q5'] = ["By increase the min_count to five, prevent discarding too many instances that can turn out to be similar",
                 mse5]


# In[89]:


assert len(answers['Q5']) == 2
assert type(answers['Q5'][0]) == str
assertFloat(answers['Q5'][1])


# In[90]:


f = open("answers_hw4.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




