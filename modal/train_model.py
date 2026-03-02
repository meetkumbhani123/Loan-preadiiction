import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_and_save():
    print("Loading data...")
    # Go up one directory to find 'data'
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Loan_default.csv')
    df = pd.read_csv(data_path)

    if 'LoanID' in df.columns:
        df.drop('LoanID', axis=1, inplace=True)

    print("Handling missing values...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    if 'Default' in num_cols:
        num_cols = num_cols.drop('Default')
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna('Unknown', inplace=True)

    print("Encoding and splitting data...")
    X = df.drop('Default', axis=1)
    y = df['Default']
    
    # Store original categorical/numeric structure to help the API later
    meta = {
        'num_cols': list(num_cols),
        'cat_cols': list(cat_cols),
        'cat_classes': {col: df[col].unique().tolist() for col in cat_cols}
    }
    joblib.dump(meta, os.path.join(os.path.dirname(__file__), 'metadata.joblib'))

    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    joblib.dump(X_encoded.columns.tolist(), os.path.join(os.path.dirname(__file__), 'model_columns.joblib'))

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    print("Scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(os.path.dirname(__file__), 'scaler.joblib'))

    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    print(f"Train Accuracy: {model.score(X_train_scaled, y_train):.4f}")
    print(f"Test Accuracy: {model.score(X_test_scaled, y_test):.4f}")

    print("Saving model...")
    joblib.dump(model, os.path.join(os.path.dirname(__file__), 'logistic_model.joblib'))
    print("Done!")

if __name__ == '__main__':
    train_and_save()
