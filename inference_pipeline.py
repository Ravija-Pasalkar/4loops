import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load artifacts
scaler = joblib.load("saved_pipeline/scaler.pkl")
feature_columns = joblib.load("saved_pipeline/feature_columns.pkl")
label_classes = np.load("saved_pipeline/label_classes.npy", allow_pickle=True)
target_encoder = joblib.load("saved_pipeline/target_encoder.pkl")
label_encoders = joblib.load("saved_pipeline/label_encoders.pkl")
model = load_model("saved_pipeline/transformer_model.keras")

def preprocess_and_predict(input_dict):
    df = pd.DataFrame([input_dict])

    # Feature engineering (same as training)
    df["income_to_networth_ratio"] = df["annual_income"] / (df["net_worth"] + 1e-6)
    df["adjusted_debt_to_income"] = df["debt_to_income_ratio"] * df["annual_income"]
    df["investment_savings_ratio"] = df["portfolio_value"] / (df["savings_rate"] + 1e-6)

    df["annual_income"] = df["annual_income"].clip(lower=0)
    df["net_worth"] = df["net_worth"].clip(lower=0)

    df["age_group"] = pd.cut(df["age"], bins=[18, 35, 55, np.inf], labels=["Young", "Mid-age", "Senior"])
    df["income_group"] = pd.cut(df["annual_income"], bins=[0, 50000, 150000, np.inf], labels=["Low", "Medium", "High"])
    df["net_worth_level"] = pd.cut(df["net_worth"], bins=[0, 50000, 200000, np.inf], labels=["Poor", "Stable", "Wealthy"])

    df["total_financial_score"] = df["financial_knowledge_score"] + df["macroeconomic_score"] + df["sentiment_index"]
    df["total_allocation_pct"] = df["equity_allocation_pct"] + df["fixed_income_allocation_pct"]

    if 'month' in df.columns:
        df['month'] = pd.to_datetime(df['month'], errors='coerce')
        df['month_num'] = df['month'].dt.month
        df['year'] = df['month'].dt.year
        df.drop(columns='month', inplace=True)

    # Label encode
    for col, le in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError as e:
                raise ValueError(f"Unknown category in column '{col}': {e}")

    df = df.fillna(df.mean(numeric_only=True))

    # Ensure column order
    df = df.reindex(columns=feature_columns)

    # Scale + reshape
    X_scaled = scaler.transform(df)
    X_reshaped = X_scaled.reshape(1, 1, X_scaled.shape[1])

    # Predict
    pred_idx = np.argmax(model.predict(X_reshaped), axis=-1)[0]
    return label_classes[pred_idx]
