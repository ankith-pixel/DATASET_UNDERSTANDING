import pandas as pd
import numpy as np 

# Load dataset
df = pd.read_csv("titanic.csv")

# Identify numeric and categorical columns based on dtype
numeric_col = df.select_dtypes(include=["int64","float64"]).columns.tolist()
cato_col = df.select_dtypes(include=["object"]).columns.tolist()


# ------------------------------------------------------------
# Function to classify numeric columns into meaningful groups
# ------------------------------------------------------------
def classify_numeric(s: pd.Series):
    # Get unique values (ignoring missing)
    unique_vals = s.dropna().unique()
    n_unique = len(unique_vals)

    # Case 1: Only 2 unique values → behave like binary feature
    if n_unique == 2:
        return "binary"
    
    # Case 2: If 90% of values are unique → column behaves like an ID (useless)
    # Example: PassengerId, Ticket Number, Name hash etc.
    if n_unique > 0.9 * len(s):
        return "90% of values are unique"
    
    # Case 3: Integer column with few unique values (<20) → discrete numeric
    # Example: SibSp (0–8), Parch (0–6)
    if np.issubdtype(s.dtype, np.integer) and n_unique < 20:
        return "Discrete Numeric Columns"
    
    # Otherwise → treat as continuous numeric
    return "numeric_continuous"


# Apply classifier to each numeric column
numeric_map = {col: classify_numeric(df[col]) for col in numeric_col}

# Print classification results
for col, kind in numeric_map.items():
    print(f"{col:12} -> {kind}")


# ------------------------------------------------------------
# Function to classify categorical/text-like columns
# ------------------------------------------------------------
def classify_cato(s: pd.Series):
    n = len(s)
    n_unique = s.nunique(dropna=True)
    ratio_unique = n_unique / n   # proportion of unique values

    # If more than 80% values are unique → looks like a text field or ID column
    # Example: Name, Cabin, Ticket
    if ratio_unique > 0.8:
        return "text_like"
    
    # Otherwise treat as normal categorical feature
    return "categorical"


# Apply classification to all categorical columns
cato_map = {col: classify_cato(df[col]) for col in cato_col}

# Print classification results
for col, kind in cato_map.items():
    print(f"{col:10} -> {kind}")
