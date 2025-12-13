import pandas as pd
import numpy as np
from twokenize import tokenize
from sklearn.feature_extraction.text import CountVectorizer
import string
from gensim.models import Word2Vec
import gensim
import nltk
from nltk import pos_tag
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from sklearn.feature_extraction import DictVectorizer
nltk.download('averaged_perceptron_tagger_eng')
from scipy.sparse import hstack
import spacy
nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("../data/NewsCategorizer.csv")

def sub_leaves(tree, label):
    return [
        subtree.leaves()
        for subtree in tree.subtrees()
        if isinstance(subtree, nltk.Tree) and subtree.label() == label
    ]

def extract_ner_counts(tree, target_labels):
    counts = {f"NER_{lbl}": 0 for lbl in target_labels}

    for subtree in tree.subtrees():
        if isinstance(subtree, nltk.Tree):
            lbl = subtree.label()
            if lbl in target_labels:
                counts[f"NER_{lbl}"] += 1

    return counts


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

#tri-grams feature
trigram_counts = {}
for words in df_filt["tokenized_headline"]:
    for i in range(len(words) - 2):
        trigram = (words[i], words[i+1], words[i+2])
        if trigram in trigram_counts:
            trigram_counts[trigram] += 1
        else:
            trigram_counts[trigram] = 1

# NER feature extraction
entity_labels = ["PERSON", "ORG", "GPE", "MONEY", "LOC", "FAC", "NORP"]
ner_data = []
for doc in nlp.pipe(df_filt["headline"], batch_size=1000):
    counts = {"NER_PERSON": False, "NER_ORG": False, "NER_GPE": False, "NER_MONEY": False, "NER_LOC": False, "NER_FAC": False, "NER_NORP": False}
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "MONEY", "LOC", "FAC", "NORP"):
            counts[f"NER_{ent.label_}"] = True
    ner_data.append(counts) 

# Create bigram_counts per headline (Sparse matrix - (row, col), num of bigrams count)
texts = [" ".join(word) for word in df_filt["tokenized_headline"]]
vectorizer = CountVectorizer(ngram_range=(2,2), max_features=5000)
bigrams_data = vectorizer.fit_transform(texts)

#trigram counts
vectorizer_tri = CountVectorizer(ngram_range=(3,3), max_features=5000)
trigrams_data = vectorizer_tri.fit_transform(texts)


# unigram counts
# Unigrams (single words)
vectorizer_uni = CountVectorizer(ngram_range=(1,1), max_features=5000)
unigrams_data = vectorizer_uni.fit_transform(texts)


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
        if tag in counts:
            counts[tag] += 1
        else:
            counts[tag] = 1
    pos_data.append(counts)


# Sparse matrix POS (row, num of unique tags)
vec = DictVectorizer()
pos_data = vec.fit_transform(pos_data)

#ner features
ner_matrix = vec.fit_transform(ner_data)

# assign a sentiment number for each headline (we can likely use the compound number)
sentiments = []

for headline in df_filt["tokenized_headline"]:
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment = sentiment_analyzer.polarity_scores(" ".join(headline))
    sentiments.append(sentiment)

# Final data/features: ngrams_data, word2vec_data, pos_data, sentiments
unigrams_dense = unigrams_data.toarray()
bigrams_dense = bigrams_data.toarray()
pos_dense = pos_data.toarray()
sentiment_dense = np.array([s['compound'] for s in sentiments]).reshape(-1, 1)
ner_dense = ner_matrix.toarray()
trigrams_dense = trigrams_data.toarray()

# Stack all features horizontally
X = np.hstack([unigrams_dense, bigrams_dense, word2vec_data, pos_dense, sentiment_dense, ner_dense,
                trigrams_dense])

# Target labels
y = df_filt['category'].values

headlines = df_filt['headline'].values

# print(X)
# print(X.shape)
