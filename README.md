# Titanic Survival Predictor: Defensive ML Pipeline

A high-performance Random Forest pipeline built with defensive engineering principles. This project achieves **79% Accuracy** on the Titanic dataset using automated feature engineering and hyperparameter tuning.

## 🏗 Project Structure
- `data/`: Raw dataset storage.
- `src/preprocessing.py`: Modular cleaning, title extraction, and categorical encoding.
- `src/train_model.py`: Automated GridSearch tuning and feature importance analysis.
- `src/predict.py`: Inference script for real-time survival probability.
- `titanic_model.pkl`: Serialized production-ready model.

## 🛠 Features & Engineering
- **Defensive Pathing:** Scripts are casing-agnostic and handle dynamic directory resolution.
- **Title Extraction:** Engineered social status from passenger names to impute missing age data.
- **Tuned Random Forest:** Optimized via 5-fold cross-validation.

## 🚀 Usage
1. Install dependencies: \`pip install -r requirements.txt\`
2. Train the model: \`python src/train_model.py\`
3. Run a prediction: \`python src/predict.py\`

## 📊 Feature Importance
1. Sex/Title (Mr) ~40%
2. Fare (Class) ~17%
3. Age ~15%
