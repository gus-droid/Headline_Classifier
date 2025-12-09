import sys
#from porter_stemmer import PorterStemmer
#from BitVector import BitVector
#from classify_util import *
import string
from data import df_filt
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from data import X, y, headlines
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

# split data
headlines_train, headlines_test, X_train, X_test, y_train, y_test = train_test_split(headlines, X, y, test_size=0.2)


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
print(precision_score(y_test, preds, average='macro'))
print(recall_score(y_test, preds, average='macro'))
print(f1_score(y_test, preds, average=None))

# ConfusionMatrix
confusionmatrix = confusion_matrix(y_test, preds, labels = ["POLITICS", "WORLD NEWS", "ENTERTAINMENT", "BUSINESS", "TRAVEL"])

print(confusionmatrix)

mis_idx = np.where(y_test != preds)[0]

# X_test_raw MUST be the original list/array of texts aligned to X_test
mis_texts = [headlines_test[i] for i in mis_idx]
mis_true  = y_test[mis_idx]
mis_pred  = preds[mis_idx]

for t, true, pred in zip(mis_texts, mis_true, mis_pred):
    print(f"TRUE: {true} | PRED: {pred} | TEXT: {t}")

# Best Model:
# 0.824
# [[748  83  51  84  26]
# [ 77 802  26  48  47]
# [ 38  28 858  29  51]
# [ 44  31  25 857  42]
# [ 29  49  33  39 855]]