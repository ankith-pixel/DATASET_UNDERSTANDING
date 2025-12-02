------------------------------------------------------------
CORRELATION (PEARSON) numerical feature -> numberical target
------------------------------------------------------------

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic.csv")

numeric_col = df.select_dtypes(include=["int64", "float64"]).columns
cato_col = df.select_dtypes(include=["object"]).columns

cor_matrix = df[numeric_col].corr()
#print(cor_matrix)
target_corr = cor_matrix['Survived'].sort_values(ascending=False)
print(target_corr)

# plt.figure(figsize=(8,6))
# sns.heatmap(cor_matrix, annot=True, fmt=".2f", cmap="coolwarm")
# plt.show()

df_encode = df.copy()

df_encode['sex_encode'] = df_encode['Sex'].map({'male':0 , 'female':1})

numerical_col2 = df_encode.select_dtypes(include=["int64", "float64"]).columns

cor_matrix2 = df_encode[numerical_col2].corr()
print(cor_matrix2['Survived'].sort_values(ascending=False))


---------------------------------------------------
ANOVA(f-test) numeric feature -> catorigical target
---------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import f_oneway

df = pd.read_csv("titanic.csv")

def run_anova(feature):
    survived = df[df['Survived']==1][feature].dropna()
    not_survived = df[df['Survived']==0][feature].dropna()

    f , p = f_oneway(survived, not_survived)

    print(f"features: {feature}")
    print(f"f_value {f}")
    print( f"p_value {p}")

    if p < 0.05:
        print("Strong feature")
    else:
        print("weak feature")

    print("-" *50)

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

df = pd.read_csv("titanic.csv")

def run_chi_square(feature):
    table = pd.crosstab(df[feature], df['Survived'])
    chi2 , p, dof, expected = chi2_contingency(table)

    print(f"Feature {feature}")
    print(f"Chi- square value {chi2}")
    print(f"p- value {p}")
    print(f"Degree of freedom {dof}")
    print(f"table value {table}")
    print(f"expected value {expected}")

    if p < 0.05:
        print("strong feature(KEEP)")
    else:
        print("weak feature(DROP)")

    print("-" * 50)

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

df = pd.read_csv("titanic.csv")

df_encode = df.copy()

for col in df_encode.select_dtypes(include=["object"]).columns:
    df_encode[col] = df_encode[col].astype('category').cat.codes

x = df_encode.drop(columns=['Survived']).dropna()
y = df_encode['Survived'].dropna()

mi_score = mutual_info_classif(x,y, discrete_features='auto')
mi_score = pd.Series(mi_score, index = x.columns)
mi_score = mi_score.sort_values(ascending=False)

print(mi_score)
