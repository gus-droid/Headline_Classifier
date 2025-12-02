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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Naive Bayes Training (# Initial Output: 0.558)
#model1 = GaussianNB()
#model1.fit(X_train, y_train)
#y_pred = model1.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)


# Logistic Regression:
model2 = LogisticRegression(max_iter=2000, multi_class="multinomial")
model2.fit(X_train, y_train)
preds = model2.predict(X_test)
print(accuracy_score(y_test, preds))

# ConfusionMatrix
confusionmatrix = confusion_matrix(y_test, preds, labels = ["POLITICS", "WORLD NEWS", "ENTERTAINMENT", "BUSINESS", "TRAVEL"])

print(confusionmatrix)

# Best Model:
# 0.6618
# [[563 144  86 110  66]
#  [ 92 660  84  99  86]
#  [ 59  68 731  60  80]
#  [ 72  88  51 651 149]
#  [ 35  70  80 112 704]]