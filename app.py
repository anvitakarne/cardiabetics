import streamlit as st
import time
import numpy as np
import pandas as pd
import joblib

# Set Streamlit page config
st.set_page_config(
    page_title="Cardiabetics",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="auto"
)

# ECG background CSS
st.markdown("""
    <style>
        body {
            background-image: url("https://raw.githubusercontent.com/anvitakarne/cardiabetics/main/ecg-pattern-light.png");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }
    </style>
""", unsafe_allow_html=True)

# Load models and scalers
diabetes_model = joblib.load("diabetes_model.pkl")
diabetes_scaler = joblib.load("scaler.pkl")

heart_model = joblib.load("heart_model.pkl")
heart_scaler = joblib.load("heart_scaler.pkl")

# ---- Splash Screen ----
def show_splash():
    splash_placeholder = st.empty()
    with splash_placeholder.container():
        st.markdown("""
            <div style="text-align:center;">
                <img src="https://raw.githubusercontent.com/anvitakarne/cardiabetics/main/logo.png" width="220">
                <h1 style="font-size: 2.6rem; margin-bottom: 0;">Cardiabetics</h1>
                <p style="font-size: 1.1rem; margin-top: 0;">Your ML-powered health companion for predicting Diabetes & Heart Disease</p>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(3)
        splash_placeholder.empty()

# Show splash screen only once
if 'splash_shown' not in st.session_state:
    show_splash()
    st.session_state.splash_shown = True

# ---- Sidebar ----
st.sidebar.markdown("""
### ℹ️ About
Predict your risk of **Diabetes** or **Heart Disease** based on simple health metrics using trained ML models.

**Developed by:** Anvita  
**App Name:** Cardiabetics  
**Tech Stack:** Streamlit · scikit-learn · Python
""")

# ---- App Title ----
st.markdown("""
    <div style="text-align:center;">
        <h1 style="font-size: 2.4rem;">🩺 Disease Risk Score Calculator</h1>
        <p style="font-size: 1.1rem;">A machine learning-powered health companion</p>
    </div>
""", unsafe_allow_html=True)

# ---- Prediction Type ----
prediction_type = st.selectbox("🎯 Choose Prediction Type", ["Diabetes", "Heart Disease"])

# ---- Diabetes Prediction Form ----
if prediction_type == "Diabetes":
    st.markdown("""### 📝 Enter your health details:""", unsafe_allow_html=True)

    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 200, 100)
    blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 10, 120, 30)

    if st.button("🔍 Predict Diabetes Risk"):
        input_data = diabetes_scaler.transform([[
            pregnancies, glucose, blood_pressure, skin_thickness, insulin,
            bmi, dpf, age
        ]])
        prediction = diabetes_model.predict_proba(input_data)[0][1]

        if prediction > 0.5:
            st.error(f"⚠️ High risk of diabetes. (Risk Score: {prediction:.2f})")
            st.markdown("💡 Consider scheduling a check-up with your doctor.")
        else:
            st.success(f"✅ Low risk of diabetes. (Risk Score: {prediction:.2f})")
            st.markdown("🎉 Keep up the healthy habits!")

# ---- Heart Disease Prediction Form ----
elif prediction_type == "Heart Disease":
    st.markdown("""### 📝 Enter your health details (for Heart Disease Risk):""", unsafe_allow_html=True)

    age = st.number_input("Age", 20, 120, 50)
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
    oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
    ca = st.number_input("Number of Major Vessels (0-3)", 0, 3, 0)

    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    restecg = st.selectbox("Resting ECG", ["normal", "ST-T abnormality", "left ventricular hypertrophy"])
    slope = st.selectbox("ST Slope", ["upsloping", "flat", "downsloping"])
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

    if st.button("🪀 Predict Heart Disease Risk"):
        # Convert categorical inputs to encoded features
        sex_Male = 1 if sex == "Male" else 0
        cp_typical = cp == "typical angina"
        cp_atypical = cp == "atypical angina"
        cp_non_anginal = cp == "non-anginal"

        restecg_normal = restecg == "normal"
        restecg_abnormal = restecg == "ST-T abnormality"

        slope_up = slope == "upsloping"
        slope_flat = slope == "flat"

        thal_normal = thal == "normal"
        thal_reversible = thal == "reversable defect"

        input_data = heart_scaler.transform([[
            age, trestbps, chol, thalach, oldpeak, ca,
            sex_Male, cp_atypical, cp_non_anginal, cp_typical,
            restecg_normal, restecg_abnormal,
            slope_flat, slope_up,
            thal_normal, thal_reversible
        ]])

        prediction = heart_model.predict_proba(input_data)[0][1]

        if prediction > 0.5:
            st.error(f"⚠️ High risk of heart disease. (Risk Score: {prediction:.2f})")
            st.markdown("💡 Please consult a cardiologist.")
        else:
            st.success(f"✅ Low risk of heart disease. (Risk Score: {prediction:.2f})")
            st.markdown("❤️ Your heart is doing great — keep it up!")
