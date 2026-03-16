# utils.py -- robust model loader and prediction utilities
import os, joblib, numpy as np, pandas as pd
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "model", "model_data.pkl")
model_data = joblib.load(MODEL_PATH)
model = model_data.get("model")
scaler = model_data.get("scaler")
features = model_data.get("features")
columns_to_scale = model_data.get("cols_to_scale")
def data_preparation(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income,
                     loan_amount, loan_tenure_months, total_loan_months,
                     loan_purpose, loan_type, residence_type):
    data_input = {
        'age': age,'avg_dpd_per_dm': avg_dpd_per_dm,'credit_utilization_ratio': credit_utilization_ratio,
        'dmtlm': dmtlm,'income': income,'loan_amount': loan_amount,
        'lti': loan_amount / income if income > 0 else 0,
        'total_loan_months': total_loan_months,'loan_tenure_months': loan_tenure_months,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0
    }
    df = pd.DataFrame([data_input])
    if columns_to_scale and scaler is not None:
        try:
            df[columns_to_scale] = scaler.transform(df[columns_to_scale])
        except Exception:
            pass
    if features:
        try:
            df = df[features]
        except Exception:
            pass
    return df
def calculate_credit_score(input_df, base_score=300, scale_length=600):
    if model is None:
        raise RuntimeError("Model not loaded.")
    default_probability = model.predict_proba(input_df)[:, 1]
    non_default_probability = 1 - default_probability
    credit_score = base_score + non_default_probability.flatten() * scale_length
    score_val = int(credit_score.flatten()[0])
    if 300 <= score_val < 500: rating = 'Poor'
    elif 500 <= score_val < 650: rating = 'Average'
    elif 650 <= score_val < 750: rating = 'Good'
    elif 750 <= score_val <= 900: rating = 'Excellent'
    else: rating = 'Undefined'
    return float(default_probability.flatten()[0]), score_val, rating
def predict(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income,
            loan_amount, loan_tenure_months, total_loan_months,
            loan_purpose, loan_type, residence_type):
    input_df = data_preparation(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income,
                                loan_amount, loan_tenure_months, total_loan_months,
                                loan_purpose, loan_type, residence_type)
    return calculate_credit_score(input_df)
