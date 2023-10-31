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

### Outlier Removal using z-score and Retraining

- **Models with No Outliers**:
- Decision Tree Classifier: Mean Accuracy ~ 87.09%, Standard Deviation: ~0.14%
- Naive Bayes Classifier: Mean Accuracy ~ 84.70%, Standard Deviation: ~0.23%

### Retraining with Top 10 Features

- **Before Outlier Removal**: 
  - Decision Tree Classifier: Mean Accuracy ~ 86.37%, Standard Deviation ~ 0.13%
  - Naive Bayes Classifier: Mean Accuracy ~ 85.22%, Standard Deviation ~ 0.19%
  
- **After Outlier Removal**: 
  - Decision Tree Classifier: Mean Accuracy ~ 87.44%, Standard Deviation ~ 0.13%
  - Naive Bayes Classifier: Mean Accuracy ~ 86.00%, Standard Deviation ~ 0.25%

### Ensemble Modeling
To further improve model performance, an ensemble method was employed. This combined two different classifiers: Decision Tree and Naive Bayes. A soft voting strategy was used, giving more weight to the Decision Tree model with weights [2, 1] for Decision Tree and Naive Bayes, respectively.

Despite the combined approach, the ensemble model did not result in a significant improvement in performance. The 10-fold cross-validation mean accuracy remained at approximately 87.44% with a standard deviation of ~ 0.13%. 

### Hyper-Parameter Tuning

To fine-tune the models, hyper-parameter tuning was performed using the Optuna framework. Optuna was configured to run 50 trials to search for the best hyperparameters for both Decision Tree and Naive Bayes classifiers. The objective was to maximize the mean accuracy of a 10-Fold Stratified Cross-Validation.

#### Tuning Strategy

For Decision Tree Classifier:
- `criterion`: ['gini', 'entropy']
- `max_depth`: Range from 10 to 50
- `min_samples_split`: Range from 2 to 15
- `min_samples_leaf`: Range from 1 to 10

For Naive Bayes Classifier:
- `var_smoothing`: Logarithmically spaced values ranging from 1e-10 to 1e-2

#### Best Results

After 50 trials, the best performing model and hyperparameters were:
- **Best Model**: Decision Tree Classifier
- **Best Parameters**: 
  - `criterion`: gini
  - `max_depth`: 47
  - `min_samples_split`: 5
  - `min_samples_leaf`: 8
- **Best Score**: 90.19% Accuracy I found finally. 


---

## Deployment in Streamlit <a name="streamlit"></a>

- The model is deployed in a Streamlit app where users can input the feature values to get a prediction.
- Only top 10 important features are considered for prediction to simplify the user interface.

---

## File Directory

- `EDA.ipynb`: Notebook containing Exploratory Data Analysis.
- `Predictive_Modeling_Classification.ipynb`: Notebook containing the predictive models.
- `README.md`: This README file.
- `best_dt_model.pkl`: Saved model checkpoint for the best-performing Decision Tree Classifier.
- `label_encoders.pkl`: Saved label encoders for categorical variables.
- `requirements.txt`: Text file containing the libraries required for Streamlit app.
- `scaler.pkl`: Saved scaler for feature scaling.
- `streamlit_app.py`: Streamlit app for the project.

Each file has a specific role in the project and contributes to either the analysis, modeling, or deployment phase.



