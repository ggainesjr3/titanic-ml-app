from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Titanic Survival Predictor")

# main.py

# 1. Load the entire dictionary from the file
model_artifact = joblib.load('models/titanic_model.pkl')

# 2. Extract the individual pieces using the keys we saved in train_model.py
model = model_artifact['model']
model_features = model_artifact['features']

# Define the structure of the incoming data
class Passenger(BaseModel):
    Pclass: int
    Sex: str  # "male" or "female"
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str  # "S", "C", or "Q"
    Title: str  # "Mr", "Miss", "Mrs", "Master", or "Rare"

@app.post("/predict")
def predict_survival(passenger: Passenger):
    # 1. Convert Pydantic object to a dictionary
    input_data = passenger.dict()
    
    # 2. Calculate engineered features
    input_data['FamilySize'] = input_data['SibSp'] + input_data['Parch'] + 1
    input_data['IsAlone'] = 1 if input_data['FamilySize'] == 1 else 0
    
    # 3. Handle categorical encoding (Sex)
    input_data['Sex'] = 1 if input_data['Sex'].lower() == 'male' else 0
    
    # 4. Handle One-Hot Encoding (Title & Embarked)
    # We create a DataFrame with 0s and then set the specific categories to 1
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df, columns=['Title', 'Embarked'])
    
    # 5. Reindex to match the model's training columns
    # This fills any missing dummy columns (like Title_Rare) with 0
    df = df.reindex(columns=model_features, fill_value=0)
    
    # 6. Make Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return {
        "survived": bool(prediction),
        "survival_probability": round(float(probability), 2)
    }
