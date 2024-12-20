import streamlit as st
import pickle
import pandas as pd

# Load the model and pipeline
model_path = "loan_status_model.pkl"
with open(model_path, "rb") as f:
    pipeline = pickle.load(f)

# Streamlit UI for user inputs
st.title("Loan Approval Prediction")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self-Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])

# Convert inputs into a DataFrame
input_data = pd.DataFrame({
    "Gender": [gender],
    "Married": [married],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_amount_term],
    "Credit_History": [1 if credit_history == "Yes" else 0],
    "Property_Area": [property_area],
    "Dependents": [dependents]
})

# Apply the pipeline to preprocess inputs
processed_input = pipeline.named_steps["preprocessor"].transform(input_data)

# Make prediction
if st.button("Predict"):
    prediction = pipeline.named_steps["classifier"].predict(processed_input)
    if prediction[0] == 1:
        st.success("The loan is likely to be approved.")
    else:
        st.error("The loan is likely to be denied.")
