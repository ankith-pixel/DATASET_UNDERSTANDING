import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv("titanic.csv")

# numeric_col = df.select_dtypes(include=["int64", "float64"]).columns
# for col in numeric_col:
#     plt.figure(figsize=(6,4))
#     sns.histplot(df[col], kde=True)
#     plt.title(f"Distribution of {col}")
#     plt.show()
# print(df[col].describe)

cato_col = df.select_dtypes(include=["object"]).columns

for col in cato_col:
    print(df[col].value_counts(dropna=True))
    print("/n")
    plt.figure(figsize=(6,4))
    sns.countplot(x=df[col])
    plt.title(f"Distribution on {col}")
    plt.show()

print(df[col].describe)
