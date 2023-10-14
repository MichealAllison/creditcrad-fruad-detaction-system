import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the pre-trained model
model = LogisticRegression()  # Load your trained model here

# Web app
st.set_page_config(page_title="Credit Card Fraud Detection System")
st.title("Credit Card Fraud Detection System")

# User input
input_data = st.text_input('Enter the values separated by commas (e.g., val1, val2, val3)')

if input_data:
    input_values = [float(val.strip()) for val in input_data.split(',')]

    if len(input_values) != len(X_train.columns):
        st.error(f"Please provide {len(X_train.columns)} values separated by commas.")
    else:
        # Make a prediction
        prediction = model.predict([input_values])

        if prediction[0] == 0:
            st.write("Legitimate transaction")
        else:
            st.write("Fraudulent transaction")

# Add more sections, explanations, and visual enhancements here
