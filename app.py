import streamlit as st
import pickle
import pandas as pd

# Load pipeline
with open("churn_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Customer Churn Prediction")

st.write("Enter customer details:")

# Inputs
input_data = {
    "CreditScore": st.number_input("Credit Score", 300, 900, 650),
    "Age": st.number_input("Age", 18, 100, 35),
    "Tenure": st.number_input("Tenure", 0, 10, 5),
    "Balance": st.number_input("Balance", 0.0, 100000.0, 50000.0),
    "NumOfProducts": st.number_input("Products", 1, 4, 1),
    "HasCrCard": st.selectbox("Has Credit Card", [0, 1]),
    "IsActiveMember": st.selectbox("Active Member", [0, 1]),
    "EstimatedSalary": st.number_input("Salary", 0.0, 200000.0, 50000.0),
    "Geography": st.selectbox("Geography", ["France", "Germany", "Spain"]),
    "Gender": st.selectbox("Gender", ["Male", "Female"])
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer likely to CHURN\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Customer NOT likely to churn\nProbability: {probability:.2f}")