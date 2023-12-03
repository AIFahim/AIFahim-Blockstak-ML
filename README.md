# AIFahim-Blockstak-ML

# Bank Telemarketing Prediction Model
<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/web-ui.png" width="600" height="500">
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

**Summary Statistics**:
- Age: The average age is about 41 years. The youngest person is 19 and the oldest is 87.
- Job: The most common job type is "management".
- Marital Status: Most people are married.
- Education: Most people have a "secondary" level of education.
- Default: Very few people have credit in default. Most have "no" in this column.
- Balance: The average bank balance is about 1423 Euros. Some have a negative balance, while the highest is 71,188 Euros.
- Housing: Most people have a housing loan.
- Loan: Most people do not have a personal loan.
- Contact: The most common method of contact is "cellular".
- Day: Contacts are made throughout the month, with the average day being around the 16th.
- Month: May is the most common month for contact.
- Duration: The average duration of contact is about 264 seconds.
- Campaign: On average, people are contacted about 3 times during a campaign.
- Pdays: Most people haven't been contacted before.
- Previous: Most people have zero contacts before the current campaign.
- Poutcome: The outcome of the previous campaign is mostly "unknown".
- Y (Target): Most people have not subscribed to a term deposit.

**Missing Values**: No missing values in the dataset.
  
**Categorical Features**: 
- Job Distribution: Most people work in management, blue-collar jobs, or are technicians.
- Marital Status Distribution: The majority are married.
- Education Level Distribution: Most have a secondary level of education.
- Contact Method Distribution: Cellular is the most common method of contact.
- Last Contact Month Distribution: Most contacts were made in the month of May.
- Outcome of Previous Campaign: For most people, the outcome of the previous campaign is unknown.
- Term Deposit Subscription: A smaller number of people have subscribed to a term deposit compared to those who haven't.

<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/categoricals_1.png" width="400" height="300">
<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/categoricals_2.png" width="400" height="300">
<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/categoricals_3.png" width="400" height="300">
<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/categoricals_4.png" width="400" height="300">
<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/categoricals_5.png" width="400" height="300">
<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/categoricals_6.png" width="400" height="300">
<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/categoricals_7.png" width="400" height="300">


**Numerical Features**: 
- Age Distribution: Most people are between 30 and 40 years old.
- Balance Distribution: Most people have a balance between 0 and 5000 Euros, but there are some outliers with much higher balances.
- Contact Duration Distribution: Most of the contact durations are less than 500 seconds.
- Number of Contacts in Campaign: Most people have been contacted less than 10 times in the current campaign.
- Days Since Last Contact: Most people have not been contacted before (indicated by the spike at -1).
- Number of Contacts Before Campaign: Most people were not contacted before the current campaign.

<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/numericals_1.png" width="400" height="300">
<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/numericals_2.png" width="400" height="300">
<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/numericals_3.png" width="400" height="300">
<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/numericals_4.png" width="400" height="300">
<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/numericals_5.png" width="400" height="300">
<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/numericals_6.png" width="400" height="300">


**Correlation between numerical variables**:
  **Observations**
- `pdays` and `previous` have a correlation of 0.45, which is moderate. This suggests that if a customer was contacted before, it's likely that more days have passed since the last contact.
- `duration` and `campaign` have a slight negative correlation of -0.08, suggesting that more contacts in the campaign could slightly decrease the duration of calls. However, this is not a strong correlation.
- `age` has very low correlation with other numerical variables, indicating that it might not be a strong predictor for other numerical variables in this dataset.

<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/corr_numericals.png" width="400" height="300">


**Correlation of numeric features with the label**:
  **Observations**
- Duration: It has the highest positive correlation (0.40) with the label. This suggests that longer call durations are somewhat associated with a higher likelihood of a client subscribing to a term deposit.
- Previous: This variable has a correlation of 0.12 with the label, which is not very strong but still positive.
- Pdays: It has a correlation of 0.10, which is also positive but not very strong.
- Age and Balance: These have very weak positive correlations with the label.
- Day and Campaign: These have weak negative correlations with the label.

<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/corr_w_labels.png" width="400" height="300">
 
**Outliers**: 
 **Based on Z-Scores**
 
- Z-Scores of Age: Most Z-scores are around 0, with a few reaching up to 3 or more.
- Z-Scores of Balance: Most Z-scores are below 3, but there are some that go beyond.
- Z-Scores of Day: All Z-scores are well within the -3 to 3 range.
- Z-Scores of Duration: Some Z-scores go beyond the 3 threshold.
- Z-Scores of Campaign: Most Z-scores are below 3, but there are some that go beyond.
- Z-Scores of Pdays: Most Z-scores are below 3, but there are some that go beyond.
- Z-Scores of Previous: Most Z-scores are below 3, but there are some that go beyond.

<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/zscores.png" width="400" height="700">


 **Based on Box-Plots**
  
- Boxplot of Age: A few outliers above the upper whisker.
- Boxplot of Balance: Several outliers, mainly on the higher end.
- Boxplot of Day: No noticeable outliers. Boxplot of Duration: Some outliers on the higher end.
- Boxplot of Campaign: Several outliers above the upper whisker.
- Boxplot of Pdays: Many outliers, mostly on the higher end.
- Boxplot of Previous: Several outliers above the upper whisker.

<img src="https://github.com/AIFahim/AIFahim-Blockstak-ML/blob/master/images/Box_plots.png" width="400" height="700">

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

The model is deployed in a Streamlit app, which provides an interactive user interface for making predictions. Only the top 10 important features are considered in the prediction model to make the user interface simpler and more focused.

Deployed model here: [Streamlit App](https://aifahim-blockstak-ml-ntwnox2nctw8qbaukqrmoy.streamlit.app/)



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



