import pandas as pd
import numpy as np
import sys
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import run_full_preprocessing

def train():
    path = 'data/train.csv'
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return

    print("--- Loading & Processing ---")
    df = pd.read_csv(path)
    df_processed = run_full_preprocessing(df)
    
    target = 'survived'
    X = df_processed.drop(columns=[target])
    y = df_processed[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tuning Grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 7, 10, None],
        'min_samples_split': [2, 5]
    }

    print("--- Running Grid Search ---")
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # EVALUATE
    y_pred = best_model.predict(X_val)
    print(f"\nBest Params: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("\nReport:\n", classification_report(y_val, y_pred))

    # --- TACTICAL FEATURE IMPORTANCE ---
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n--- Tactical Feature Importance ---")
    for f in range(X.shape[1]):
        print(f"{f + 1}. {X.columns[indices[f]]}: {importances[indices[f]]:.4f}")

    joblib.dump(best_model, 'titanic_model.pkl')
    print("\nSuccess: Tuned model and importance metrics ready.")

if __name__ == "__main__":
    train()
