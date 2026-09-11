
"""
Created on Thu Sep 10 20:44:10 2026

Author: martin

Task: Web Development using streamlit and spyder
"""


# Import libraries

import streamlit as st
import pandas as pd
import joblib

# Load joblib file which contains our model

model = joblib.load("model.joblib")

# Add title and instructions

st.title("Purchase Prediction Model ABCD")
st.subheader("Enter customer information and submit for likelihood purchase")

# Age input form
age = st.number_input(
    label = "01. Enter the customer's age",
    min_value = 18,
    max_value = 120,
    value = 35)
# Gender input form
gender = st.radio("02. Enter the customer's age",
    options = ["M", "F"])

# Credit input form
credit_score = st.number_input("03. Enter customer's credit score",
                               min_value = 0,
                               max_value = 1000,
                               value = 500)

# submit inputs to model
if st.button("Submit For Prediction"):
    
    # store the data in a dataframe for prediction
    new_data = pd.DataFrame({"age" : [age], "gender" : [gender], "credit_score" : [credit_score]})
    

    # apply model pipeline to input data and extract probability prediction
    pre_proba = model.predict_proba(new_data)[0][1]
    
    # output the prediction
    st.subheader(f"Based on this customer attribute, the model predics a purchase of {pre_proba:.0%}")