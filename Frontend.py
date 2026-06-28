import streamlit as st
import pandas as pd
import numpy as np
import joblib

#LOAD MODEL
model = joblib.load("thyroid__model.pkl")
le = joblib.load("label__encoder.pkl")
# features = joblib.load("features.pkl") 
FINAL_FEATURES =  [
    "TSH","TT4", "T4U", "T3",
    "age", "sex", "on_thyroxine"
]

st.title("🧠 Thyroid Disease Prediction System")
st.write("Enter patient lab test values:")


#INPUTS

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100)
    sex = st.selectbox("Sex", ["Male", "Female"])

    TSH = st.number_input("TSH (µIU/mL)", 0.0, 50.0)

    T3_value = st.number_input("T3 value", 0.0, 300.0)
    T3_unit = st.selectbox("T3 unit", ["nmol/L", "ng/dL", "ng/mL"])

with col2:
    test_type = st.radio("Available Tests", ["TT4 + T4U", "Only T4"])

    if test_type == "TT4 + T4U":
        TT4_value = st.number_input("TT4 value", 0.0, 50.0)
        TT4_unit = st.selectbox("TT4 unit", ["µg/dL", "nmol/L"])
        T4U = st.number_input("T4U (index 0.8 - 1.3 normal)", 0.0, 5.0)

    elif test_type == "Only T4":
        T4_value = st.number_input("T4 (Total T4)", 0.0, 50.0)
        TT4_unit = st.selectbox("T4 unit", ["µg/dL", "nmol/L"])
        T4U = 1.0   # default

    on_thyroxine = st.selectbox("On Thyroxine?", ["No", "Yes"])
    goitre = st.selectbox("Goitre?", ["No", "Yes"])

#ENCODING

sex = 1 if sex == "Male" else 0
on_thyroxine = 1 if on_thyroxine == "Yes" else 0
goitre = 1 if goitre == "Yes" else 0

#UNIT CONVERSION 

# T3 → nmol/L
if T3_unit == "nmol/L":
    T3 = T3_value
elif T3_unit == "ng/dL":
    T3 = T3_value / 65
elif T3_unit == "ng/mL":
    T3 = T3_value * 1.536

# TT4 → µg/dL
if test_type == "TT4 + T4U":
    raw_T4 = TT4_value
else:
    raw_T4 = T4_value

if TT4_unit == "µg/dL":
    TT4 = raw_T4
else:
    TT4 = raw_T4 / 12.87

if test_type == "Only T4":
    st.info("ℹ️ T4U assumed as 1.0 (normal)")

#FEATURE ENGINEERING 

# T3_T4_ratio = T3 / (TT4 + 1e-6)
FTI = TT4 / (T4U + 1e-6)

#PREDICT

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "TSH": TSH,
    "FTI": FTI,
    "TT4": TT4,
    "T4U": T4U,
    "T3": T3,
    "age": age,
    "sex": sex,
    "on_thyroxine": on_thyroxine
    }])

    #CLEANING
    # input_data = input_data.replace([np.inf, -np.inf], np.nan)
    # input_data = input_data.fillna(0)

    #FEATURE ORDER FIX
    input_data = input_data[FINAL_FEATURES]

    #PREDICTION
    y_prob = model.predict_proba(input_data)
    y_pred = np.argmax(y_prob, axis=1)

    result = le.inverse_transform(y_pred)[0]
    confidence = np.max(y_prob)

    #OUTPUT
    st.success(f"🩺 Diagnosis: {result}")
    st.info(f"Confidence: {confidence*100:.2f}%")

    # Extra info
    st.info(f"T3 (nmol/L): {T3:.2f}")
    st.info(f"TT4 (µg/dL): {TT4:.2f}")

    # Clinical hint layer
    if TSH < 0.4:
        st.warning("⚠️ Low TSH (Possible Hyperthyroid)")
    elif TSH > 4:
        st.warning("⚠️ High TSH (Possible Hypothyroid)")
