# app.py
import os
import streamlit as st
import numpy as np
import joblib

# Load model
@st.cache_resource
def load_xgb_model():
    model_path = os.path.join(os.path.dirname(__file__), "xgboost_soc_model.h5")
    return joblib.load(model_path)

model = load_xgb_model()

# UI
st.title("Battery SOC Prediction 🔋")
st.markdown("Enter battery input parameters to predict the **State of Charge (SOC)**.")

# Input: 7 feature values
feature_names = ["Voltage", "Current", "Temperature", "Battery_type_Fresh", "Battery_type_Aged", "Time", "Energy"]
user_inputs = []

cols = st.columns(2)
for i, name in enumerate(feature_names):
    col = cols[i % 2]
    val = col.number_input(f"{name}", value=0.0, step=0.1, format="%.2f")
    user_inputs.append(val)

# Predict button
if st.button("Predict SOC"):
    try:
        input_array = np.array(user_inputs).reshape(1, -1)
        prediction = model.predict(input_array)
        st.success(f"🔹 Predicted SOC: **{prediction[0]:.2f}%**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")
