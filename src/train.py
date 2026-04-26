import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# 1. SETUP PATHS
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '..', 'train.csv')
model_output_path = os.path.join(base_path, '..', 'titanic_model.pkl')

def load_and_prep():
    """Step 1: Prep ingredients. Normalizes casing for robustness."""
    if not os.path.exists(data_path):
        print(f"--- Error: File not found at {data_path} ---")
        return None, None
    
    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]
    
    if 'survived' not in df.columns:
        print("--- Error: 'survived' column not found! ---")
        return None, None

    # --- Feature Engineering ---
    if 'name' in df.columns:
        df['title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        rare = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        df['title'] = df['title'].replace(rare, 'Other').replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs')
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
        df['title'] = df['title'].map(title_mapping).fillna(0)
    else:
        df['title'] = 0

    df['age'] = df['age'].fillna(df['age'].median()).clip(lower=0, upper=80)
    if 'sex' in df.columns:
        df['sex'] = df['sex'].map({'female': 1, 'male': 0}).fillna(0)
    else:
        df['sex'] = 0

    features = ['pclass', 'sex', 'age', 'title']
    valid_features = [f for f in features if f in df.columns]
    X = df[valid_features]
    y = df['survived']
    
    return X, y

def train_production_model():
    """Step 2: Train and serialize the model."""
    print("--- Starting Production Training Cycle ---")
    X, y = load_and_prep()
    
    if X is None or y is None:
        print("--- Training Aborted: Check data columns. ---")
        return

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, model_output_path)
    print(f"--- SUCCESS: Model saved to {model_output_path} ---")

if __name__ == "__main__":
    train_production_model()