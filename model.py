import sys
#from porter_stemmer import PorterStemmer
#from BitVector import BitVector
#from classify_util import *
import string
from data import df_filt
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from data import X, y
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import accuracy_score


# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Initial Output: 0.558