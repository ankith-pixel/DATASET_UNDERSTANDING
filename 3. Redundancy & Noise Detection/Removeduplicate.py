import numpy as np
import pandas as pd

df = pd.read_csv('StudentPerformanceFactors.csv')

# # -------------------
# # DUPLICATE ROWS
# # -------------------

df_clean = df.T.duplicated()
df_cleaned = df.columns[df_clean]
print( "Duplicate Rows: ", df_cleaned.tolist() )

# # -------------------
# # HIGH CORRELATION
# # -------------------

cor_matrix = df.corr(numeric_only=True)
high_cor = []

for col1 in cor_matrix.columns:
    for col2 in cor_matrix.columns:
        if col1 > col2:
            if abs(cor_matrix.loc[col1,col2]) > 0.9:
                high_cor.append((col1, col2, cor_matrix.loc[col1, col2]))

print(high_cor)

# # -------------------
# # Constant COLUMNS
# # -------------------

constant_cols =  [col for col in df.columns if df[col].nunique(dropna=False) ==1]
print("Constant Columns: ", constant_cols)

# # -----------------------
# # Nearly Constant COLUMNS
# # -----------------------

near_constant_col= []
threshold = 0.98

for col in df.columns:
    if df[col].dropna().empty:
        continue   
    top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]

    if top_freq >= threshold:
        near_constant_col.append((col, top_freq))
for col , near_freq in near_constant_col:
    print(col)

# # ---------------------------------
# # Features with high missing values
# # ---------------------------------

missing_ratio = df.isnull().mean()
missing_threshold = 0.4
high_missing = missing_ratio[missing_ratio > missing_threshold]
print("Features with high missing values:", high_missing.index.tolist())

# # --------------------
# # Low Variance COLUMNS
# # --------------------

low_variance_cols = []
num_col = df.select_dtypes(include=["int64", "float64"]).columns

for col in num_col:
    var = df[col].var()
    low_variance_cols.append((col, var))

low_variance_cols = [col for col, var in low_variance_cols if var < 0.1]
print("Low Variance Columns:", low_variance_cols)


