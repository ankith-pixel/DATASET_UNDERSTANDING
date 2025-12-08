import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("titanic.csv")

# -------------------------------------------------------------
# Plot distributions for all categorical columns
# -------------------------------------------------------------

# Select only object-type (categorical/text) columns
cato_col = df.select_dtypes(include=["object"]).columns

for col in cato_col:

    # Print frequency count of each category (ignoring NaN)
    # Helps understand class imbalance, rare categories, typos, etc.
    print(df[col].value_counts(dropna=True))
    print("\n")   # newline for readability

    # Plot category distribution
    plt.figure(figsize=(6, 4))

    # Countplot → shows how many times each category appears
    sns.countplot(x=df[col])

    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)  # rotate text for readability
    plt.tight_layout()       # prevent label cutoff
    plt.show()

# Print basic summary statistics for the last column processed
print(df[col].describe)
