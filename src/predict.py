import pandas as pd
import joblib
import sys
import os

# Ensure we can import our preprocessing logic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import run_full_preprocessing

def predict_passenger(passenger_data):
    # 1. Load the trained model
    model_path = 'titanic_model.pkl'
    if not os.path.exists(model_path):
        print("Error: Model file not found. Run train_model.py first.")
        return

    model = joblib.load(model_path)
    
    # 2. Convert raw input to DataFrame
    df_input = pd.DataFrame([passenger_data])
    
    # 3. Use our existing pipeline to clean the data
    # This handles the lowercase, titles, and one-hot encoding automatically
    df_processed = run_full_preprocessing(df_input)
    
    # 4. Get the model's training features to ensure alignment
    model_features = model.feature_names_in_
    
    # 5. Reindex to match model features (adds missing columns as 0)
    df_final = df_processed.reindex(columns=model_features, fill_value=0)
    
    # 6. Predict
    prediction = model.predict(df_final)[0]
    probability = model.predict_proba(df_final)[0][1]
    
    status = "SURVIVED" if prediction == 1 else "PERISHED"
    print(f"\n--- Prediction Result ---")
    print(f"Status: {status}")
    print(f"Survival Probability: {probability:.2%}")

if __name__ == "__main__":
    # Test Data: Let's test a hypothetical First Class passenger
    test_passenger = {
        'Pclass': 1,
        'Name': 'Gaines, Mr. Gary',
        'Sex': 'male',
        'Age': 28,
        'SibSp': 0,
        'Parch': 0,
        'Fare': 150.0,
        'Embarked': 'S'
    }
    
    predict_passenger(test_passenger)
