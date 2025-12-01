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

print(X)
print(X.shape)
print(y)
print(y.shape)
