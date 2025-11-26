# 📊 DATASET UNDERSTANDING

<p align="center">
  <img src="https://github.com/user-attachments/assets/f0e17f18-9cd4-4a0a-9181-898f3f451a33" width="500" />
</p>

---

## **1. Data Type Understanding**

Before any analysis, identify the type of each feature:

- **Numeric continuous** → age, salary  
- **Numeric discrete** → counts  
- **Categorical nominal** → color  
- **Categorical ordinal** → low/medium/high  
- **Text**  
- **Binary** → 0/1  
- **ID-like** → user_id (usually useless)

This helps determine the correct EDA methods, preprocessing steps, and encodings.

---

## **2. Exploratory Data Analysis (EDA)**

### **A. Distribution Analysis**
- **Numeric:** histograms, boxplots  
- **Categorical:** value counts  
Used to detect:
- Skewness  
- Outliers  
- Rare categories  
- Invalid values  

### **B. Interaction With Target**
Check how each feature affects the label:

- [**Correlation**](https://medium.com/@abdallahashraf90x/all-you-need-to-know-about-correlation-for-machine-learning-e249fec292e9) → numerical → numerical
- [**ANOVA**](https://medium.com/data-science/anova-for-feature-selection-in-machine-learning-d9305e228476) → categorical feature → numerical target  
- [**Chi-Square**](https://medium.com/data-science/chi-square-test-for-feature-selection-in-machine-learning-206b1f0b8223) → categorical → categorical  
- [**Mutual Information**](https://medium.com/@suvendulearns/decoding-mutual-information-mi-a-guide-for-machine-learning-practitioners-b0f0ca0b30c9) → works for any feature type  

This reveals predictive features vs useless ones.

---

## **3. Redundancy & Noise Detection**

✔ **Duplicate features**  
Examples:  
- `age` and `age_in_years`  
- `salary` and `income`  

✔ **High-cardinality categorical features**  
Example: ZIP code with 500+ unique values → high risk of overfitting.

✔ **Constant or near-constant features**  
- Same value in 99% of rows → drop

✔ **Too many missing values**  
- If >60–70% missing → usually remove

---

## **4. Simple Statistical Tests**

Use these to measure how strongly each feature influences the target:

- **ANOVA** → categorical feature vs numeric target  
- **Chi-Square Test** → categorical vs categorical  
- **Mutual Information** → general dependence (nonlinear, any type)

---

## **5. Ask Data Owner / Read Documentation**

If something looks suspicious or unclear:  
- Confirm with domain experts  
- Read dataset documentation  

You avoid incorrect assumptions and data leakage.

---
