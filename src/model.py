import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. LOAD DATA (Defensive Loading)
# This logic ensures the script finds train.csv regardless of where you run it from
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'train.csv')

try:
    df = pd.read_csv(data_path)
    print("--- Success: Data loaded into the prep station ---")
except FileNotFoundError:
    print(f"--- Error: train.csv not found at {data_path} ---")
    print("Check your ~/titanic-ml-project folder!")
    exit()

# 2. THE PREP STATION (Feature Engineering)
def refine_titanic_data(data):
    """
    Cleans the data and creates new features. 
    Similar to a bar's prep shift: organizing ingredients before service.
    """
    df_refined = data.copy()
    
    # Title Extraction: Organizing passengers by 'Seating Class'
    df_refined['Title'] = df_refined['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Grouping rare titles to prevent 'noise'
    rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df_refined['Title'] = df_refined['Title'].replace(rare_titles, 'Other')
    df_refined['Title'] = df_refined['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df_refined['Title'] = df_refined['Title'].replace('Mme', 'Mrs')
    
    # Age Clipping: Setting 'Max Capacity' to handle outliers
    df_refined['Age'] = df_refined['Age'].fillna(df_refined['Age'].median())
    df_refined['Age'] = df_refined['Age'].clip(lower=0, upper=80)
    
    # Mapping Categories to Numbers (Machine Learning speaks in numbers)
    df_refined['Sex'] = df_refined['Sex'].map({'female': 1, 'male': 0})
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
    df_refined['Title'] = df_refined['Title'].map(title_mapping).fillna(0)
    
    # Select only the features that matter for the final 'drink'
    features = ['Survived', 'Pclass', 'Sex', 'Age', 'Title']
    return df_refined[features]

# Process the data
processed_df = refine_titanic_data(df)

# 3. THE SERVICE SHIFT (Model Training)
X = processed_df.drop('Survived', axis=1)
y = processed_df['Survived']

# Splitting 80% for training, 20% for a 'Secret Shopper' test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest: Think of this as a 'Committee of Experts' making the decision
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 4. THE QUALITY AUDIT (Evaluation)
y_pred = model.predict(X_test)

print("\n=== PORTFOLIO QUALITY REPORT ===")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nBreakdown of Precision/Recall (The 'Pour Cost' check):")
print(classification_report(y_test, y_pred))

# Feature Importance: Which 'Ingredient' was most important?
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- Key Predictors ---")
print(importances)