import pandas as pd 
df=pd.read_csv("fake-reviews.csv")
print(df.head())
# Quick look at the first rows

print(df.head())

# Count missing values in each column

print(df.isnull().sum())

# Character length of each review

df["char_length"] = df["text_"].apply(len)

# Word count of each review

df["word_count"] = df["text_"].str.split().apply(len)

import seaborn as sns # To visualize data

import matplotlib.pyplot as plt # To manipulate graphics of data

sns.boxplot(x="char_length", y="label", data=df)

plt.title("Character Length of Reviews by Label")

plt.xlabel("Character length")

plt.ylabel("Review label")

plt.show()



