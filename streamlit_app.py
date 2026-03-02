import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Loan Default Prediction", page_icon="💸", layout="centered")

@st.cache_resource
def load_models():
    # Load from the modal directory
    model = joblib.load('modal/logistic_model.joblib')
    scaler = joblib.load('modal/scaler.joblib')
    model_columns = joblib.load('modal/model_columns.joblib')
    metadata = joblib.load('modal/metadata.joblib')
    return model, scaler, model_columns, metadata

try:
    model, scaler, model_columns, metadata = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Make sure you have trained the model first.")
    st.stop()

st.title("💸 AI Loan Risk Assessment Dashboard")
st.markdown("Enter the applicant's details below to get an instant ML-powered prediction on loan default risk.")

# Create form
with st.form("prediction_form"):
    st.subheader("Applicant Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        income = st.number_input("Annual Income ($)", min_value=0, value=75000)
        loan_amount = st.number_input("Loan Amount ($)", min_value=100, value=20000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720)
        months_employed = st.number_input("Months Employed", min_value=0, value=48)
        num_credit_lines = st.number_input("Num Credit Lines", min_value=0, value=3)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=5.5)
        loan_term = st.number_input("Loan Term (Months)", min_value=1, value=36)
        
    with col2:
        dti_ratio = st.number_input("DTI Ratio", min_value=0.0, value=0.35)
        
        # Categorical Inputs dynamically loaded from the trained metadata
        education = st.selectbox("Education", metadata['cat_classes'].get('Education', ["High School", "Bachelor's", "Master's", "PhD"]))
        employment_type = st.selectbox("Employment Type", metadata['cat_classes'].get('EmploymentType', ["Full-time", "Part-time", "Self-employed", "Unemployed"]))
        marital_status = st.selectbox("Marital Status", metadata['cat_classes'].get('MaritalStatus', ["Single", "Married", "Divorced"]))
        has_mortgage = st.selectbox("Has Mortgage", metadata['cat_classes'].get('HasMortgage', ["Yes", "No"]))
        has_dependents = st.selectbox("Has Dependents", metadata['cat_classes'].get('HasDependents', ["Yes", "No"]))
        loan_purpose = st.selectbox("Loan Purpose", metadata['cat_classes'].get('LoanPurpose', ["Business", "Home", "Auto", "Education", "Other"]))
        has_cosigner = st.selectbox("Has Co-Signer", metadata['cat_classes'].get('HasCoSigner', ["Yes", "No"]))
        
    submit_button = st.form_submit_button(label="Run Risk Analysis", use_container_width=True)

if submit_button:
    # Prepare the input dictionary
    input_dict = {
        'Age': age, 'Income': income, 'LoanAmount': loan_amount, 'CreditScore': credit_score,
        'MonthsEmployed': months_employed, 'NumCreditLines': num_credit_lines, 
        'InterestRate': interest_rate, 'LoanTerm': loan_term, 'DTIRatio': dti_ratio,
        'Education': education, 'EmploymentType': employment_type, 'MaritalStatus': marital_status,
        'HasMortgage': has_mortgage, 'HasDependents': has_dependents, 'LoanPurpose': loan_purpose,
        'HasCoSigner': has_cosigner
    }
    
    # Convert input to DataFrame
    df_input = pd.DataFrame([input_dict])
    
    # Initialize a dataframe with zeros for all model columns
    df_processed = pd.DataFrame(columns=model_columns)
    df_processed.loc[0] = 0.0 
    
    # Fill numerical values
    for col in metadata['num_cols']:
        if col in model_columns:
            df_processed.at[0, col] = df_input.at[0, col]
            
    # Fill categorical values (dummy encoding)
    for col in metadata['cat_cols']:
        val = df_input.at[0, col]
        dummy_col = f"{col}_{val}"
        if dummy_col in model_columns:
            df_processed.at[0, dummy_col] = 1.0
            
    # Scale the row
    X_scaled = scaler.transform(df_processed)
    
    # Predict
    prob = model.predict_proba(X_scaled)[0][1]
    prediction = int(model.predict(X_scaled)[0])
    
    st.markdown("---")
    st.subheader("Prediction Results")
    
    # Display Results visually
    col1, col2 = st.columns(2)
    
    col1.metric("Default Risk Probability", f"{prob * 100:.1f}%")
    
    if prediction == 1:
        col2.error("🚨 HIGH RISK: The model predicts this loan is likely to default.")
        st.progress(float(prob), text="Risk Level")
    else:
        col2.success("✅ LOW RISK: The model predicts this loan is likely to be repaid.")
        st.progress(float(prob), text="Risk Level")
