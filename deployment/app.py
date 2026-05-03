import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Download and load the trained model
model_path = hf_hub_download(
    repo_id="SRKiran/tourism_model",
    filename="best_tourism_model_v1.joblib"
)
model = joblib.load(model_path)

# Label encoding mappings (sorted alphabetically, matching prep.py LabelEncoder)
encoders = {
    'TypeofContact':  sorted(["Company Invited", "Self Enquiry"]),
    'Occupation':     sorted(["Salaried", "Free Lancer", "Small Business", "Large Business", "SE"]),
    'Gender':         sorted(["Male", "Female"]),
    'MaritalStatus':  sorted(["Single", "Married", "Divorced", "Unmarried"]),
    'Designation':    sorted(["Executive", "Manager", "Senior Manager", "AVP", "VP"]),
    'ProductPitched': sorted(["Basic", "Deluxe", "King", "Standard", "Super Deluxe"]),
}

def label_encode(col, value):
    return encoders[col].index(value)

# Streamlit UI
st.title("Wellness Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to **purchase the Wellness Tourism Package**
based on their demographic and interaction details.
Please fill in the customer details below to get a prediction.
""")

st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business", "SE"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income (INR)", min_value=1000, max_value=100000, value=20000, step=500)
    passport = st.selectbox("Holds Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    own_car = st.selectbox("Owns a Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_children_visiting = st.number_input("Number of Children Visiting (< 5 yrs)", min_value=0, max_value=5, value=0)
    preferred_star = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
    num_trips = st.number_input("Number of Trips per Year", min_value=1, max_value=20, value=3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    num_followups = st.number_input("Number of Follow-ups", min_value=1, max_value=10, value=3)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)

# Assemble input into DataFrame with label-encoded categorical values
input_data = pd.DataFrame([{
    'Age':                      age,
    'TypeofContact':            label_encode('TypeofContact', type_of_contact),
    'CityTier':                 city_tier,
    'DurationOfPitch':          duration_of_pitch,
    'Occupation':               label_encode('Occupation', occupation),
    'Gender':                   label_encode('Gender', gender),
    'NumberOfPersonVisiting':   num_person_visiting,
    'NumberOfFollowups':        num_followups,
    'ProductPitched':           label_encode('ProductPitched', product_pitched),
    'PreferredPropertyStar':    preferred_star,
    'MaritalStatus':            label_encode('MaritalStatus', marital_status),
    'NumberOfTrips':            num_trips,
    'Passport':                 passport,
    'PitchSatisfactionScore':   pitch_satisfaction,
    'OwnCar':                   own_car,
    'NumberOfChildrenVisiting': num_children_visiting,
    'Designation':              label_encode('Designation', designation),
    'MonthlyIncome':            monthly_income,
}])

# Predict button
if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"This customer is **likely to purchase** the Wellness Tourism Package. (Confidence: {probability:.1%})")
    else:
        st.warning(f"This customer is **unlikely to purchase** the Wellness Tourism Package. (Confidence: {1 - probability:.1%})")
