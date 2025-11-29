import pandas as pd
import numpy as np
from twokenize import tokenize
from sklearn.feature_extraction.text import CountVectorizer
import string
from gensim.models import Word2Vec
import gensim
import nltk
from nltk import pos_tag
from sklearn.feature_extraction import DictVectorizer
nltk.download('averaged_perceptron_tagger_eng')

df = pd.read_csv("data/NewsCategorizer.csv")

# filtering/cleaning data 
df = df[["headline", "category"]]
df["tokenized_headline"] = df["headline"].apply(tokenize)

for h in range(len(df["tokenized_headline"])):
    for w in df["tokenized_headline"][h]:
        if w in ["VIDEO", "PHOTO", "PHOTOS", "QUIZ", "GRAPHIC", "WATCH"]:
            df["tokenized_headline"][h].remove(w)
        if w in string.punctuation and w not in ["$", "?", "!"]: #maybe add more 
            df["tokenized_headline"][h].remove(w)

df["tokenized_headline"] = df["tokenized_headline"].apply(lambda x: [i.lower() for i in x])   

df_filt = df[df["category"].isin(["POLITICS", "WORLD NEWS", "ENTERTAINMENT", "BUSINESS", "TRAVEL"])]


# n-grams feature
# Bigram counts for entire dataset
bigram_counts = {}
for words in df_filt["tokenized_headline"]:
    for i in range(len(words) - 1):
        bigram = (words[i], words[i+1])
        if bigram in bigram_counts:
            bigram_counts[bigram] += 1
        else:
            bigram_counts[bigram] = 1

# Create bigram_counts per headline (Sparse matrix - (row, col), num of bigrams count)
texts = [" ".join(word) for word in df_filt["tokenized_headline"]]
vectorizer = CountVectorizer(ngram_range=(2,2))
ngrams_data = vectorizer.fit_transform(texts)


# Word2Vec
word2vecmodel = gensim.models.Word2Vec(df_filt["tokenized_headline"], min_count=1, vector_size=100, window=2)
headline_vectors = []

# Average word embeddings for headlines
for word_lst in df_filt["tokenized_headline"]:
    vectors = [word2vecmodel.wv[word] for word in word_lst]
    headline_vec = np.mean(vectors, axis=0)
    headline_vectors.append(headline_vec)

word2vec_data = np.vstack(headline_vectors)


# POS-tagging
pos_data = []
for word_lst in df_filt["tokenized_headline"]:
    pos_tags = pos_tag(word_lst)
    counts = {}
    for word, tag in pos_tags:
        if tag in counts == 1:
            counts[tag] += 1
        else:
            counts[tag] = 1
    pos_data.append(counts)

# Sparse matrix POS (row, num of unique tags)
vec = DictVectorizer()
pos_data = vec.fit_transform(pos_data)


# Final data/features: ngrams_data, word2vec_data, pos_data
