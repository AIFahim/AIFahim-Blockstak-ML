import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained Decision Tree model
model = joblib.load('best_dt_model.pkl')

# Load the dictionary of label encoders
label_encoders = joblib.load('label_encoders.pkl')

# Load the pre-trained StandardScaler
scaler = joblib.load('scaler.pkl')

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
    month = st.sidebar.selectbox('Month', label_encoders['month'].classes_)
    pdays = st.sidebar.number_input('Days since last contact', min_value=-1)
    job = st.sidebar.selectbox('Job Type', label_encoders['job'].classes_)
    poutcome = st.sidebar.selectbox('Previous Outcome', label_encoders['poutcome'].classes_)
    campaign = st.sidebar.slider('Number of Contacts in Campaign', 1, 63, 2)
    education = st.sidebar.selectbox('Education', label_encoders['education'].classes_)
    

    # Label encode categorical features
    user_data = {
        'duration': [duration],
        'balance': [balance],
        'age': [age],
        'day': [day],
        'month': [int(label_encoders['month'].transform([month])[0])],
        'pdays': [pdays],
        'job': [int(label_encoders['job'].transform([job])[0])],
        'poutcome': [int(label_encoders['poutcome'].transform([poutcome])[0])],
        'campaign': [campaign],
        'education': [int(label_encoders['education'].transform([education])[0])],
    }


    # Display the user_data dictionary in the Streamlit app
    st.write("User inputted data: ", user_data)
    
    # Convert to DataFrame to match the input shape of our model
    user_data_df = pd.DataFrame(user_data)

    # Display the DataFrame for debug
    st.write("User inputted data before scaling: ", user_data_df)
    
    # Scale the user data
    user_data_scaled = scaler.transform(user_data_df)
    
    # Display the scaled data
    st.write("User inputted data after scaling: ", user_data_scaled)

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
