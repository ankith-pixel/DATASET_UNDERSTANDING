import pandas as pd
import numpy as np 
df = pd.read_csv("titanic.csv")
#print(df.info()) 
numeric_col = df.select_dtypes(include=["int64","float64"]).columns.tolist()
cato_col = df.select_dtypes(include=["object"]).columns.tolist()
# print(numeric_col)
# print(cato_col)
def classify_numeric(s: pd.Series):
    unique_vals = s.dropna().unique()
    n_unique = len(unique_vals)

    if n_unique == 2:
        return "binary"
    
    if n_unique > 0.9 * len(s):
        return "90% of values are unique"
    
    if np.issubdtype(s.dtype, np.integer) and n_unique < 20:
        return "Discrete Numeric Columns"
    
    return "numeric_continuous"

numeric_map = {col: classify_numeric(df[col]) for col in numeric_col}
for col , kind in numeric_map.items():
    print(f" {col:12} -> {kind}")

def classify_cato(s: pd.Series):
    n = len(s)
    n_unique = s.nunique(dropna=True)
    ratio_unique = n_unique/n

    if ratio_unique > 0.8:
        return "text_like"
    
    return "categorical"

cato_map = {col: classify_cato(df[col]) for col in cato_col}
for col , kind in cato_map.items():
    print(f" {col:10} -> {kind}")