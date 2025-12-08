import numpy as np
import pandas as pd

df = pd.read_csv('StudentPerformanceFactors.csv')

# # -------------------
# # DUPLICATE ROWS
# # -------------------
# Transpose the dataframe so rows → columns, then check duplicated columns.
# This indirectly identifies duplicate rows in original df.
df_clean = df.T.duplicated()

# Extract names of duplicate rows (columns in transposed form)
df_cleaned = df.columns[df_clean]
print("Duplicate Rows: ", df_cleaned.tolist())

# # -------------------
# # HIGH CORRELATION
# # -------------------
# Compute correlation matrix only for numeric columns
cor_matrix = df.corr(numeric_only=True)

high_cor = []

# Compare each pair of columns only once
for col1 in cor_matrix.columns:
    for col2 in cor_matrix.columns:
        # Avoid repeating same pair and avoid self-correlation
        if col1 > col2:
            # If absolute correlation > 0.9 → highly correlated (redundant)
            if abs(cor_matrix.loc[col1, col2]) > 0.9:
                high_cor.append((col1, col2, cor_matrix.loc[col1, col2]))

print(high_cor)

# # -------------------
# # Constant COLUMNS
# # -------------------
# Column is constant if it has exactly one unique value.
constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
print("Constant Columns: ", constant_cols)

# # -----------------------
# # Nearly Constant COLUMNS
# # -----------------------
# Column is nearly constant if one value appears ≥98% of the time.

near_constant_col = []
threshold = 0.98

for col in df.columns:
    # Skip empty columns
    if df[col].dropna().empty:
        continue
    
    # Find frequency of the most common value
    top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]

    # If top value dominates the column → mark as near-constant
    if top_freq >= threshold:
        near_constant_col.append((col, top_freq))

# Print only column names
for col, near_freq in near_constant_col:
    print(col)

# # ---------------------------------
# # Features with high missing values
# # ---------------------------------
# Calculate percentage of missing values per column
missing_ratio = df.isnull().mean()

missing_threshold = 0.4  # 40% missing allowed
high_missing = missing_ratio[missing_ratio > missing_threshold]

# Print column names with high missing values
print("Features with high missing values:", high_missing.index.tolist())

# # --------------------
# # Low Variance COLUMNS
# # --------------------
# Low variance = columns where data does not change much

low_variance_cols = []
num_col = df.select_dtypes(include=["int64", "float64"]).columns

# Calculate variance for each numeric column
for col in num_col:
    var = df[col].var()
    low_variance_cols.append((col, var))

# Keep columns with variance < 0.1 → very low information
low_variance_cols = [col for col, var in low_variance_cols if var < 0.1]
print("Low Variance Columns:", low_variance_cols)
