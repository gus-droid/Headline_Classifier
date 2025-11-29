import pandas as pd;
from twokenize import tokenize
import string

df = pd.read_csv("/Users/matthewjordan/Desktop/LING2270/hw03-files/Headline_Classifier/NewsCategorizer.csv")

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




