import streamlit as st
import joblib
import numpy as np

# Load the trained Decision Tree model
model = joblib.load('best_dt_model.pkl')

# Create a sidebar for user input
st.sidebar.header('User Input')

def get_user_input():
    age = st.sidebar.slider('Age', 18, 100, 25)
    job = st.sidebar.selectbox('Job Type', ['admin', 'technician', 'blue-collar'])
    balance = st.sidebar.number_input('Balance', min_value=0)
    housing = st.sidebar.selectbox('Housing Loan', ['yes', 'no'])
    
    # Include all other features needed for prediction
    marital = st.sidebar.selectbox('Marital Status', ['single', 'married', 'divorced'])
    education = st.sidebar.selectbox('Education', ['primary', 'secondary', 'tertiary'])
    default = st.sidebar.selectbox('Has Credit in Default?', ['yes', 'no'])
    loan = st.sidebar.selectbox('Has Personal Loan?', ['yes', 'no'])
    contact = st.sidebar.selectbox('Contact Type', ['cellular', 'telephone'])
    
    # Create a NumPy array of the input features
    user_data = {
        'age': [age],
        'job': [job],
        'balance': [balance],
        'housing': [housing],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'loan': [loan],
        'contact': [contact]
    }

    return user_data

# Get user input
user_input = get_user_input()

# Display the prediction result
st.header('Prediction Result')
prediction = model.predict(user_input)

if prediction[0] == 1:
    st.write('The customer is likely to subscribe to a term deposit.')
else:
    st.write('The customer is unlikely to subscribe to a term deposit.')
