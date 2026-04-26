import streamlit as st
import pandas as pd
import joblib
import os
from preprocessing import run_full_preprocessing

# Set page config for that Brutalist/Tactical look
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Tactical Survival Monitor")
st.markdown("---")

# Load the saved model
if os.path.exists('titanic_model.pkl'):
    model = joblib.load('titanic_model.pkl')
else:
    st.error("Model file not found! Please run 'python src/train_model.py' first.")
    st.stop()

# --- SIDEBAR INPUTS ---
st.sidebar.header("Passenger Profile")
name = st.sidebar.text_input("Passenger Name", "Gaines, Mr. Gary")
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 100, 28)
pclass = st.sidebar.selectbox("Ticket Class (1st = Richest)", [1, 2, 3])
fare = st.sidebar.number_input("Fare Paid ($)", 0.0, 512.0, 150.0)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 6, 0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"])

# --- PREDICTION LOGIC ---
raw_data = {
    'Pclass': pclass, 'Name': name, 'Sex': sex, 'Age': age,
    'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': embarked
}

df_input = pd.DataFrame([raw_data])
df_processed = run_full_preprocessing(df_input)

# Align with model features
model_features = model.feature_names_in_
df_final = df_processed.reindex(columns=model_features, fill_value=0)

prediction = model.predict(df_final)[0]
probability = model.predict_proba(df_final)[0][1]

# --- RESULTS DISPLAY ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Verdict")
    if prediction == 1:
        st.success("✅ SURVIVED")
    else:
        st.error("💀 PERISHED")

with col2:
    st.subheader("Survival Chance")
    st.metric(label="Probability", value=f"{probability:.2%}")

st.progress(probability)

st.markdown("---")
st.info("Feature importance rankings suggest Sex, Title, and Fare are the primary survival drivers.")
