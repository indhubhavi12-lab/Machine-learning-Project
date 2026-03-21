import streamlit as st
import pandas as pd
import pickle
import os
from model_utils import load_pipeline, predict_customer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Churn Prediction", layout="centered")

# ---------------- LOAD MODEL ----------------
file_path = "churn_pipeline.pkl"

@st.cache_resource
def load_pipeline():
    try:
        if not os.path.exists(file_path):
            st.error("❌ Model file not found!")
            st.stop()

        if os.path.getsize(file_path) == 0:
            st.error("❌ Model file is empty!")
            st.stop()

        with open(file_path, "rb") as f:
            pipeline = pickle.load(f)

        return pipeline

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


pipeline = load_pipeline()

model = pipeline["model"]
scaler = pipeline["scaler"]
features = pipeline["features"]

# ---------------- PREDICT FUNCTION ----------------
def predict_customer(data):
    df = pd.DataFrame([data])

    for col in features:
        if col not in df:
            df[col] = 0

    df = df[features]
    df_scaled = scaler.transform(df)

    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    return pred, prob


# ---------------- UI ----------------
st.title("📊 Telecom Customer Churn Prediction")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly Charges", 0.0, 10000.0, 100.0)

with col2:
    total = st.number_input("Total Charges", 0.0, 100000.0, 1000.0)

phone = st.selectbox("Phone Service", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

# ---------------- BUTTON ----------------
if st.button("🔍 Predict"):

    data = {col: 0 for col in features}

    data["tenure"] = tenure
    data["MonthlyCharges"] = monthly
    data["avg_monthly_spend"] = total / (tenure + 1)

    data["PhoneService"] = 1 if phone == "Yes" else 0
    data[f"InternetService_{internet}"] = 1
    data[f"Contract_{contract}"] = 1
    data[f"PaymentMethod_{payment}"] = 1

    pred, prob = predict_customer(pipeline, data, features)

    if pred is not None:
        st.subheader("📢 Result")

        if pred == 1:
            st.error(f"⚠️ Customer will churn (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Customer will stay (Probability: {prob:.2f})")

        if prob > 0.7:
            st.error("🔴 High Risk")
        elif prob > 0.4:
            st.warning("🟠 Medium Risk")
        else:
            st.success("🟢 Low Risk")

        st.progress(int(prob * 100))
        st.write(f"📊 Churn Probability: {prob:.2%}")
        st.write(pred, prob)