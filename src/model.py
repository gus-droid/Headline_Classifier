import sys
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

def training(X, y, headlines):
    # split data
    headlines_train, headlines_test, X_train, X_test, y_train, y_test = train_test_split(headlines, X, y, test_size=0.2, random_state=42)


    # Naive Bayes Training (# Initial Output: 0.558)
    #model1 = GaussianNB()
    #model1.fit(X_train, y_train)
    #y_pred = model1.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)


    # Logistic Regression:
    model2 = LogisticRegression(max_iter=2000, multi_class="multinomial")
    model2.fit(X_train, y_train)
    preds = model2.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision_macro": float(precision_score(y_test, preds, average="macro")),
        "recall_macro": float(recall_score(y_test, preds, average="macro")),
        "f1_per_class": f1_score(y_test, preds, average=None).tolist()
    }

    # ConfusionMatrix
    confusionmatrix = confusion_matrix(y_test, preds, labels = ["POLITICS", "WORLD NEWS", "ENTERTAINMENT", "BUSINESS", "TRAVEL"])

    mis_idx = np.where(y_test != preds)[0]

    # print first 20 only
    print("\n=== MISCLASSIFIED EXAMPLES ===\n")
    for idx in mis_idx[:20]:
        print(f"TRUE: {y_test[idx]} | PRED: {preds[idx]}")
        print(f"HEADLINE: {headlines_test[idx]}")
        print("-" * 50)

    return metrics, confusionmatrix
