# AIFahim-Blockstak-ML

# Bank Telemarketing Prediction Model

## Table of Contents

1. [Exploratory Data Analysis (EDA)](#eda)
2. [Predictive Modeling/Classification](#predictive-modeling)
3. [Deployment in Streamlit](#streamlit)

---

## Exploratory Data Analysis (EDA) <a name="eda"></a>

### Objectives

- Understand the data better and prepare it for machine learning models.
- Perform Summary Statistics, Missing Values Check, Visualizations, Features Correlations, Features Importance, and Outlier Observations.

### Observations

- **Summary Statistics**: Most people are around 41 years old, work in management, and have an average bank balance of about 1423 Euros.
- **Missing Values**: No missing values in the dataset.
- **Categorical Features**: Most people are married, have a secondary level of education, and the most common job type is "management".
- **Numerical Features**: Most people have a balance between 0 and 5000 Euros, and the average duration of contact is about 264 seconds.
- **Correlations**: Duration has the highest positive correlation with the target label.
- **Outliers**: Some features like 'balance' and 'duration' have outliers which were handled.

---

## Predictive Modeling/Classification <a name="predictive-modeling"></a>

### Objectives

- Data Preprocessing including Label Encoding and Feature Scaling.
- Initial Modeling with 10-fold cross-validation.
- Retraining with Outlier Removed and Top Features.
- Hyperparameter Tuning and Best Model Selection.

### Initial Model Training

- Decision Tree Classifier: Mean Accuracy ~ 86.73%, Standard Deviation ~ 1.03%
- Naive Bayes Classifier: Mean Accuracy ~ 83.46%, Standard Deviation ~ 1.79%

### Outlier Removal and Retraining

- **New Models with No Outliers**: Decision Tree Classifier: Mean Accuracy ~ 87.09%, Naive Bayes Classifier: Mean Accuracy ~ 84.70%

### Retraining with Top 10 Features

- **Before Outlier Removal**: 
  - Decision Tree Classifier: Mean Accuracy ~ 86.37%, Standard Deviation ~ 0.13%
  - Naive Bayes Classifier: Mean Accuracy ~ 85.22%, Standard Deviation ~ 0.19%
  
- **After Outlier Removal**: 
  - Decision Tree Classifier: Mean Accuracy ~ 87.44%, Standard Deviation ~ 0.13%
  - Naive Bayes Classifier: Mean Accuracy ~ 86.00%, Standard Deviation ~ 0.25%

### Hyperparameter Tuning

- Best Model: Decision Tree Classifier
- Best Parameters: {'classifier': 'DecisionTree', 'criterion': 'gini', 'max_depth': 47, 'min_samples_split': 5, 'min_samples_leaf': 8}
- Best Score: 90.19%

---

## Deployment in Streamlit <a name="streamlit"></a>

- The model is deployed in a Streamlit app where users can input the feature values to get a prediction.
- Only top 10 important features are considered for prediction to simplify the user interface.

---


