import streamlit as st
import joblib
import pandas as pd
import os

# --- 1. CONFIGURATION & DATA LOADING ---
# 'wide' layout allows the charts and metrics to breathe
st.set_page_config(page_title="Titanic Predictor & Insights", page_icon="🚢", layout="wide")

@st.cache_data
def load_raw_data():
    """Loads the historical CSV for the dashboard charts."""
    return pd.read_csv('titanic.csv')

@st.cache_resource
def load_model_artifact():
    """Loads the trained model and feature list."""
    if not os.path.exists('models/titanic_model.pkl'):
        return None
    return joblib.load('models/titanic_model.pkl')

df = load_raw_data()
artifact = load_model_artifact()

# --- 2. HEADER & KPI METRICS ---
st.title("🚢 Titanic: Data Insights & Survival Prediction")
st.markdown("Explore historical trends and test individual survival probability using Machine Learning.")

# Calculate KPIs
total_passengers = len(df)
overall_survival = f"{round(df['Survived'].mean() * 100, 1)}%"
avg_fare = f"£{round(df['Fare'].mean(), 2)}"

# Display Metrics in 3 columns
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Total Passengers", total_passengers)
col_m2.metric("Overall Survival Rate", overall_survival)
col_m3.metric("Average Ticket Price", avg_fare)

st.divider()

# --- 3. DATA VISUALIZATION SECTION ---
st.header("📊 Historical Survival Trends")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Survival by Class")
    class_survival = df.groupby('Pclass')['Survived'].mean()
    st.bar_chart(class_survival)
    st.caption("1 = First Class, 2 = Second, 3 = Third. Higher class strongly correlates with survival.")

with col_b:
    st.subheader("Survival by Sex")
    sex_survival = df.groupby('Sex')['Survived'].mean()
    st.bar_chart(sex_survival)
    st.caption("Historical data shows a significantly higher survival rate for females.")

st.divider()

# --- 4. PREDICTION SECTION ---
st.header("🔮 Survival Predictor")
st.write("Enter details to see how the model weights these specific factors.")

if artifact:
    model = artifact['model']
    model_features = artifact['features']

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3])
            sex = st.selectbox("Sex", ["female", "male"])
            age = st.slider("Age", 0, 100, 30)
        with col2:
            sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
            parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
            fare = st.number_input("Fare Paid", 0.0, 512.0, 32.0)
        
        submit = st.form_submit_button("Run Prediction")

    if submit:
        # Feature Engineering (Sync with train_model.py)
        sex_numeric = 0 if sex == "female" else 1
        is_child = 1 if age < 12 else 0
        family_size = sibsp + parch
        
        # Build the 8-feature DataFrame
        input_df = pd.DataFrame([[
            pclass, sex_numeric, float(age), sibsp, parch, 
            float(fare), is_child, family_size
        ]], columns=model_features)

        prediction = model.predict(input_df)[0]
        
        st.divider()
        if prediction == 1:
            st.balloons()
            st.success("### Prediction: The passenger likely survived! 🎉")
        else:
            st.error("### Prediction: The passenger likely did not survive. 😔")
            
        with st.expander("🛠️ View Model Input Features"):
            st.table(input_df)
else:
    st.warning("Model file not found! Please run `python3 train_model.py` in your terminal to enable predictions.")
