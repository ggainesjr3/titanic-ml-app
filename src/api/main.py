import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. INITIALIZE THE APP (This must be at the top level for Uvicorn)
app = FastAPI()

# 2. SETUP PATHS
# This looks for the model in the main project folder (~/titanic-ml-project/titanic_model.pkl)
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "..", "titanic_model.pkl")

# 3. MANDATORY CORS SETUP
# This allows your React frontend (port 5173) to talk to this API (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. LOAD THE MODEL
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"--- SUCCESS: Model loaded from {model_path} ---")
else:
    model = None
    print(f"--- ERROR: Model NOT FOUND at {model_path} ---")

# 5. DEFINE DATA FORMAT
class Passenger(BaseModel):
    pclass: int
    sex: int
    age: float
    title: int

# 6. THE PREDICTION ROUTE
@app.post("/predict")
async def predict_survival(passenger: Passenger):
    if model is None:
        return {"error": "Model not loaded on server. Run train.py first."}

    # Create a DataFrame that matches the exact format used during training
    input_data = pd.DataFrame([{
        "pclass": passenger.pclass,
        "sex": passenger.sex,
        "age": passenger.age,
        "title": passenger.title
    }])

    # Generate prediction and probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "survived": int(prediction),
        "probability": float(probability)
    }

# This allows you to run it with 'python api/main.py' as well
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)