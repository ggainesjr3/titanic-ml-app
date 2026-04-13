import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV # Added GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import clean_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Added for potential use in param_grid if needed, though not strictly for these params

# 1. Load Data
df = pd.read_csv('data/train.csv')

# 2. Process Data
df_cleaned = clean_data(df)

X = df_cleaned.drop('Survived', axis=1)
y = df_cleaned['Survived']

# 3. Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- NEW: Hyperparameter Tuning ---
print("Starting hyperparameter tuning for RandomForestClassifier...")

# Define the parameter grid for RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None], # Consider 'sqrt' is common, 'log2' and None are alternatives
    'random_state': [42] # Keep random state consistent for reproducibility
}

# Initialize GridSearchCV
# Using 5-fold cross-validation. n_jobs=-1 uses all available CPU cores.
grid_search = GridSearchCV(estimator=RandomForestClassifier(),
                           param_grid=param_grid,
                           cv=5, # 5-fold cross-validation
                           scoring='accuracy', # Metric to optimize
                           n_jobs=-1, # Use all available CPU cores
                           verbose=2) # Show progress during search

# Fit GridSearchCV to find the best parameters
grid_search.fit(X_train, y_train)

# Get the best parameters and the best model estimator
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best parameters found: {best_params}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# 4. Evaluate the Best Model on the validation set
print("\n--- Evaluating the Best Model on Validation Set ---")
y_pred = best_model.predict(X_val)
print(classification_report(y_val, y_pred))

# 5. Save the Refined Model and Feature List
# Save the best model found by GridSearchCV
model_data = {
    'model': best_model,
    'features': X.columns.tolist() # Use original features from X
}
joblib.dump(model_data, 'models/titanic_model_refined.pkl') # Save with a new name to distinguish
print("Refined model and feature list saved to models/titanic_model_refined.pkl")

# 6. Visualization: Feature Importance of the BEST model
print("\nGenerating feature importance plot for the refined model...")
feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)

plt.figure(figsize=(10, 6))
feat_importances.nlargest(10).plot(kind='barh', color='skyblue')
plt.title('Top 10 Features Predicting Titanic Survival (Refined Model)')
plt.xlabel('Relative Importance Score')
plt.tight_layout()

# Save the plot
plt.savefig('models/feature_importance_refined.png')
print("Refined model feature importance plot saved to models/feature_importance_refined.png")

# No plt.show() is needed when running in a non-interactive environment.

