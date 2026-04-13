import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_titanic_model():
    # 1. Load the dataset
    # Ensure titanic.csv is in your ~/titanic-ml-project/ folder
    if not os.path.exists('titanic.csv'):
        print("Error: titanic.csv not found! Please ensure the dataset is in the project folder.")
        return

    df = pd.read_csv('titanic.csv')

    # 2. Basic Preprocessing
    # Fill missing Age with the median to avoid training errors
    df['Age'] = df['Age'].fillna(df['Age'].median())
    # Fill missing Fare with the median
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    # Encode Sex as numeric (female=0, male=1)
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

    # 3. Feature Engineering (The New Columns!)
    df['IsChild'] = (df['Age'] < 12).astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch']

    # 4. Define Features and Target
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'IsChild', 'FamilySize']
    X = df[features]
    y = df['Survived']

    # 5. Initialize and Train the Model
    print("Training the Random Forest model with 8 features...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 6. Save the Model and Feature List
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, 'titanic_model.pkl')
    
    # Save as a dictionary so app.py can easily retrieve both model and feature names
    joblib.dump({
        'model': model,
        'features': features
    }, model_path)

    print(f"✅ Success! Model saved to {model_path}")
    print(f"Features used: {features}")

if __name__ == "__main__":
    train_titanic_model()
