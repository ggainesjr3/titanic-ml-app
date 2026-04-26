import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import run_full_preprocessing

# 1. Load Data
try:
    df = pd.read_csv('data/titanic.csv')
except FileNotFoundError:
    print("❌ Error: data/titanic.csv not found.")
    exit()

# 2. Separate Target and Preprocess
y = df['Survived']
X_raw = df.drop(columns=['Survived']) 
X = run_full_preprocessing(X_raw)

# Ensure no target leakage
X.columns = [c.lower() for c in X.columns]
if 'survived' in X.columns:
    X = X.drop(columns=['survived'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. XGBoost Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

print("🚀 Training Optimized XGBoost Model...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 4. Evaluation
acc = accuracy_score(y_test, y_pred)
print(f"✅ Training Complete. Accuracy: {acc:.4f}")

# 5. Feature Importance Audit (Check for 'snitches')
importances = pd.Series(best_model.feature_importances_, index=X.columns)
print("\n--- Top 5 Predictive Features ---")
print(importances.sort_values(ascending=False).head(5))

# 6. Save Model
# XGBoost stores feature names internally if X is a DataFrame
joblib.dump(best_model, 'titanic_model.pkl')
print("\n📦 Model saved to titanic_model.pkl")
