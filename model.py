import sys
#from porter_stemmer import PorterStemmer
#from BitVector import BitVector
#from classify_util import *
import string
from data import df_filt
import pandas as pd

import nltk
from nltk.corpus import wordnet as wn

#likely missing some boilerplate  

names_url = 'https://data.cityofnewyork.us/api/views/25th-nujf/rows.csv?accessType=DOWNLOAD'
names_df = pd.read_csv(names_url)
names_df = names_df.dropna(subset=["Child's First Name"])
names_list = set(names_df["Child's First Name"].tolist())

for headline in df_filt["tokenized_headline"]:

    features = []

    for token in headline:
        if token is int or float:
            features.append("token_form=num")
        if token in names_list:
            features.append("token_form=name")
        



