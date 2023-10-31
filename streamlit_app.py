import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained Decision Tree model
model = joblib.load('best_dt_model.pkl')

# Load the dictionary of label encoders
label_encoders = joblib.load('label_encoders.pkl')

# Create a sidebar for user input
st.sidebar.header('User Input')

# Categorical options based on your data
categorical_options = {
    'job': ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'],
    'month': ['apr', 'aug', 'dec', 'feb', 'jan', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep'],
    'education': ['primary', 'secondary', 'tertiary', 'unknown'],
    'poutcome': ['failure', 'other', 'success', 'unknown'],
}

def get_user_input():
    # Collect user input for top 10 features only
    duration = st.sidebar.number_input('Duration', min_value=0)
    balance = st.sidebar.number_input('Balance', min_value=-8000, max_value=80000, value=1423)
    age = st.sidebar.slider('Age', 18, 100, 41)
    day = st.sidebar.slider('Day of Month', 1, 31, 15)
    month = st.sidebar.selectbox('Month', categorical_options['month'])
    pdays = st.sidebar.number_input('Days since last contact', min_value=-1)
    job = st.sidebar.selectbox('Job Type', categorical_options['job'])
    poutcome = st.sidebar.selectbox('Previous Outcome', categorical_options['poutcome'])
    campaign = st.sidebar.slider('Number of Contacts in Campaign', 1, 63, 2)
    education = st.sidebar.selectbox('Education', categorical_options['education'])
    

    # Label encode categorical features
    user_data = {
        'duration': [duration],
        'balance': [balance],
        'age': [age],
        'day': [day],
        'month': [label_encoders['month'].transform([month])[0]],
        'pdays': [pdays],
        'job': [label_encoders['job'].transform([job])[0]],
        'poutcome': [label_encoders['poutcome'].transform([poutcome])[0]],
        'campaign': [campaign],
        'education': [label_encoders['education'].transform([education])[0]],
    }

    # Display the user_data dictionary in the Streamlit app
    st.write("User inputted data: ", user_data)
    
    # Convert to DataFrame to match the input shape of our model
    user_data_df = pd.DataFrame(user_data)
    return user_data_df

# Get user input
user_input = get_user_input()

# Display the prediction result
st.header('Prediction Result')
prediction = model.predict(user_input)

if prediction[0] == 1:
    st.write('The customer is likely to subscribe to a term deposit.')
else:
    st.write('The customer is unlikely to subscribe to a term deposit.')
