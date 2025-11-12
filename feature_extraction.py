import string
import pandas as pd
import spacy 
from collections import Counter
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
POS_WHITELIST = {"VERB", "NOUN", "ADV"}

def pos_counts(text):

    doc = nlp(text)  # tokenizes the text

    return Counter(

        token.pos_ for token in doc if token.pos_ in POS_WHITELIST

    )  # count the number of each pos in the given tokenized text. We're only doing this for the whitelisted POS

def add_pos_features(df):

    pos_data = df["cleaned_text"].apply(pos_counts)

    pos_df = pd.DataFrame(list(pos_data)).fillna(0)  # fill null counts with 0

    pos_df.index = df.index  # align columns with original df (dataframe)

    return pd.concat([df, pos_df], axis=1)
def punctuation_count(text):
    count = 0
    for char in text:
        if char in string.punctuation: 
            count += 1
    return count
def extract_features(df, include_pos=False):
    df["char_length"]=df["cleaned_text"].apply(len)
    df["word_count"]=df["cleaned_text"].apply(lambda x: len(x.split()))
    df["punctuation_ct"] = df["cleaned_text"].apply(punctuation_count)
    df["is_extreme_star"] = df["rating"].isin([1.0, 5.0])
    if include_pos:

        df = add_pos_features(df)


    return df 