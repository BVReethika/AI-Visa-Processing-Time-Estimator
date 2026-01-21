import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Load trained model
model = joblib.load("models/random_forest_model.pkl")
# MODEL_PATH = os.path.join("models", "random_forest_model.pkl")

# if not os.path.exists(MODEL_PATH):
#     st.error("Model file not found. Please check deployment.")
#     st.stop()

# model = joblib.load(MODEL_PATH)

# import os
# import streamlit as st
# import joblib

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "random_forest_model.pkl")

# MODEL_PATH = os.path.abspath(MODEL_PATH)

# if not os.path.exists(MODEL_PATH):
#     st.error(f"Model not found at: {MODEL_PATH}")
#     st.stop()

# @st.cache_resource
# def load_model(path):
#     return joblib.load(path)

# model = load_model(MODEL_PATH)



st.title("AI Visa Processing Time Estimator")
st.write("Estimate visa processing time using AI-based prediction")

# ---------------------------
# Input Fields
# ---------------------------
applicant_country = st.selectbox(
    "Applicant Country",
    ["India", "USA", "UK", "Canada", "Australia", "Germany", "France"]
)

visa_type = st.selectbox(
    "Visa Type",
    ["Student", "Work", "Tourist"]
)

processing_center = st.selectbox(
    "Processing Center",
    ["Delhi", "Mumbai", "Chennai", "Hyderabad", "Bangalore",
    "New York", "London", "Toronto", "Berlin", "Paris"]
)

visa_status = st.selectbox(
    "Visa Status",
    ["Approved", "Rejected"]
)

application_month = st.selectbox(
    "Application Month",
    list(range(1, 13))
)

# ---------------------------
# Encode Inputs
# ---------------------------
def encode_input(value, classes):
    encoder = LabelEncoder()
    encoder.fit(classes)
    return encoder.transform([value])[0]

country_enc = encode_input(applicant_country,["India", "USA", "UK", "Canada", "Australia", "Germany", "France"])

visa_type_enc = encode_input(visa_type,["Student", "Work", "Tourist"])

center_enc = encode_input(processing_center,["Delhi", "Mumbai", "Chennai", "Hyderabad", "Bangalore","New York", "London", "Toronto", "Berlin", "Paris"])

status_enc = encode_input(visa_status,["Approved", "Rejected"])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Estimate Processing Time"):
    input_data = pd.DataFrame([{
        "applicant_country": country_enc,
        "visa_type": visa_type_enc,
        "processing_center": center_enc,
        "visa_status": status_enc,
        "application_month": application_month
    }])

    prediction = model.predict(input_data)[0]

    # Prediction range (confidence interval)
    lower = int(prediction - 5)
    upper = int(prediction + 5)

    st.success(f"Estimated Processing Time: **{lower} – {upper} days**")
    st.info("Prediction is based on historical visa processing data.")
     # Save to temporary JSON
    record = {
        "applicant_name": "Anonymous",
        "applicant_country": applicant_country,
        "visa_type": visa_type,
        "processing_center": processing_center,
        "visa_status": visa_status,
        "application_month": application_month,
        "predicted_processing_days": prediction,
        "predicted_range": [lower, upper],
        
    }

  
def add_application_temp(record):
    """
    Temporarily store application data
    """
    if "temp_applications" not in st.session_state:
        st.session_state.temp_applications = []

    st.session_state.temp_applications.append(record)

