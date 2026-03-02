from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Loan Prediction Live Dashboard")

# Determine base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODAL_DIR = os.path.join(BASE_DIR, 'modal')
STATIC_DIR = os.path.join(BASE_DIR, 'app', 'static')

# Ensure static directory exists
os.makedirs(STATIC_DIR, exist_ok=True)

# Load the saved model and preprocessors
model = joblib.load(os.path.join(MODAL_DIR, 'logistic_model.joblib'))
scaler = joblib.load(os.path.join(MODAL_DIR, 'scaler.joblib'))
model_columns = joblib.load(os.path.join(MODAL_DIR, 'model_columns.joblib'))
metadata = joblib.load(os.path.join(MODAL_DIR, 'metadata.joblib'))

class LoanApplication(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: float
    NumCreditLines: float
    InterestRate: float
    LoanTerm: float
    DTIRatio: float
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str

@app.post("/predict")
def predict_loan_default(loan: LoanApplication):
    # Convert input to DataFrame
    input_dict = loan.dict()
    df_input = pd.DataFrame([input_dict])
    
    # Initialize a dataframe with zeros for all model columns
    df_processed = pd.DataFrame(columns=model_columns)
    df_processed.loc[0] = 0.0 # Initialize with zeros
    
    # Fill numerical values
    for col in metadata['num_cols']:
        if col in model_columns:
            val = df_input.at[0, col]
            df_processed.at[0, col] = val
            
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
    
    return {
        "default_probability": float(prob),
        "prediction": prediction,
        "message": "High Risk of Default" if prediction == 1 else "Low Risk of Default"
    }

# Mount static directory for Frontend
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
