import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from preprocessing import run_full_preprocessing

st.set_page_config(page_title="Titanic Tactical Monitor V3", layout="centered")
st.title("🚢 Titanic Tactical Survival Monitor (V3.0 - Decay Logic)")
st.markdown("---")

if os.path.exists('titanic_model.pkl'):
    model = joblib.load('titanic_model.pkl')
else:
    st.error("Model file not found!"); st.stop()

# --- INPUTS ---
st.sidebar.header("Passenger Profile")
name = st.sidebar.text_input("Name", "Gaines, Mr. Gary")
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 100, 28)
pclass = st.sidebar.selectbox("Class", [1, 2, 3])
fare = st.sidebar.number_input("Fare ($)", 0.0, 512.0, 150.0)
embarked = st.sidebar.selectbox("Port", ["S", "C", "Q"])

# --- PREDICTION LOGIC ---
raw_data = {'Pclass': pclass, 'Name': name, 'Sex': sex, 'Age': age, 
            'SibSp': 0, 'Parch': 0, 'Fare': fare, 'Embarked': embarked}
df_processed = run_full_preprocessing(pd.DataFrame([raw_data]))
model_features = model.feature_names_in_
df_final = df_processed.reindex(columns=model_features, fill_value=0)

# 1. RAW INFERENCE
raw_prob = model.predict_proba(df_final)[0][1]

# 2. THE DECAY GUARDRAIL (Architect Level)
# As age exceeds 80, confidence decays exponentially
confidence = 1.0
adjusted_prob = raw_prob

if age > 80:
    # Every year over 80 reduces the survival probability by 15% compounded
    years_over = age - 80
    decay_factor = 0.85 ** years_over
    adjusted_prob = raw_prob * decay_factor
    confidence = max(0.1, 1.0 - (years_over * 0.1))

prediction = 1 if adjusted_prob >= 0.5 else 0

# --- RESULTS DISPLAY ---
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Verdict")
    st.success("SURVIVED") if prediction == 1 else st.error("PERISHED")
with c2:
    st.subheader("Probability")
    st.metric(label="Adjusted", value=f"{adjusted_prob:.1%}")
with c3:
    st.subheader("Confidence")
    st.metric(label="Data Trust", value=f"{confidence:.0%}")

st.progress(adjusted_prob)

# --- THE DYNAMIC ADVISORY ---
if confidence < 0.7:
    st.warning(f"🔍 **Low Confidence Warning:** This passenger's age ({age}) is an outlier. The model's raw probability has been decayed by {1-decay_factor:.1%} to account for lack of historical evidence.")

st.markdown("---")
st.info("Architecture V3 utilizes Exponential Decay for out-of-distribution Age handling.")
