import pickle
import os

def load_pipeline(file_path="churn_pipeline.pkl"):
    """Load saved churn model pipeline."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found!")
    with open(file_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


def predict_customer(pipeline, data, features):
    """Make churn prediction."""
    import pandas as pd

    model = pipeline["model"]
    scaler = pipeline["scaler"]

    df = pd.DataFrame([data])
    for col in features:
        if col not in df:
            df[col] = 0
    df = df[features]
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]
    return pred, prob