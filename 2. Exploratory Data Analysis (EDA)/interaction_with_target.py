------------------------------------------------------------
CORRELATION (PEARSON) numerical feature -> numberical target
------------------------------------------------------------

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic.csv")    # load dataset

numeric_col = df.select_dtypes(include=["int64", "float64"]).columns   # select all numeric columns
cato_col = df.select_dtypes(include=["object"]).columns                # select all categorical columns

cor_matrix = df[numeric_col].corr()    # compute correlation matrix between all numeric columns
#print(cor_matrix)
target_corr = cor_matrix['Survived'].sort_values(ascending=False)  # correlation of each feature with Survived target
print(target_corr)

# plt.figure(figsize=(8,6))
# sns.heatmap(cor_matrix, annot=True, fmt=".2f", cmap="coolwarm")   # heatmap for correlation visualization
# plt.show()

df_encode = df.copy()   # make a copy so original df is not modified

df_encode['sex_encode'] = df_encode['Sex'].map({'male':0 , 'female':1})  
# manually label-encode Sex to numeric so it can be used in correlation

numerical_col2 = df_encode.select_dtypes(include=["int64", "float64"]).columns  
# select updated numeric columns including sex_encode

cor_matrix2 = df_encode[numerical_col2].corr()   # recompute correlation including encoded Sex
print(cor_matrix2['Survived'].sort_values(ascending=False))   # show correlation with target



---------------------------------------------------
ANOVA(f-test) numeric feature -> catorigical target
---------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import f_oneway

df = pd.read_csv("titanic.csv")   # reload dataset

def run_anova(feature):
    # split feature values based on Survived = 1 or 0
    survived = df[df['Survived']==1][feature].dropna()
    not_survived = df[df['Survived']==0][feature].dropna()

    # run ANOVA: checks if mean of feature differs between groups
    f , p = f_oneway(survived, not_survived)

    print(f"features: {feature}")
    print(f"f_value {f}")     # higher f-value = more separation between groups
    print( f"p_value {p}")    # p-value < 0.05 means statistically significant feature

    if p < 0.05:
        print("Strong feature")   # feature affects survival
    else:
        print("weak feature")     # no strong effect

    print("-" *50)

# run ANOVA on numeric features
run_anova("Age")
run_anova("Fare")
run_anova("SibSp")
run_anova("Parch")



---------------------------------------------------------
chi-square test catorigical feature -> catorigical target 
---------------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

df = pd.read_csv("titanic.csv")   # reload dataset

def run_chi_square(feature):
    # contingency table of feature vs Survived
    table = pd.crosstab(df[feature], df['Survived'])

    # run chi-square test
    chi2 , p, dof, expected = chi2_contingency(table)

    print(f"Feature {feature}")
    print(f"Chi- square value {chi2}")   # larger value = stronger relationship
    print(f"p- value {p}")               # p < 0.05 means significant
    print(f"Degree of freedom {dof}")    # depends on categories
    print(f"table value {table}")        # actual counts
    print(f"expected value {expected}")  # expected counts if no relationship

    if p < 0.05:
        print("strong feature(KEEP)")    # reject null → they are dependent
    else:
        print("weak feature(DROP)")

    print("-" * 50)

# run tests on categorical columns
run_chi_square("Sex")
run_chi_square("Pclass")
run_chi_square("Embarked")



---------------------------------------------
Mutial Information any feature -> any target 
---------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

df = pd.read_csv("titanic.csv")   # reload dataset

df_encode = df.copy()   # make a copy for encoding

# convert categorical columns to category codes so MI can use them
for col in df_encode.select_dtypes(include=["object"]).columns:
    df_encode[col] = df_encode[col].astype('category').cat.codes

df_clear = df_encode.dropna()  # MI cannot handle NaN → remove them

x = df_clear.drop(columns=['Survived'])   # all features
y = df_clear['Survived']                  # target

# compute mutual information score
mi_score = mutual_info_classif(x,y, discrete_features='auto')

# convert result to Series for readability
mi_score = pd.Series(mi_score, index = x.columns)
mi_score = mi_score.sort_values(ascending=False)

print(mi_score)   # higher MI = more predictive power
