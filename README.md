# 📂 SYSTEM_ARCH: TITANIC_TACTICAL_MONITOR_V3
# STATUS: [DEPLOYED] | ACCURACY: [82.14%] | ENVIRONMENT: [ROSENCRANTZ]
# DEVELOPER: GARY EDWARD GAINES, JR.

---

## 🛠 TACTICAL OVERVIEW
A high-density survival prediction engine built with **XGBoost**. This system moves beyond baseline modeling by implementing **Defensive Engineering** patterns and **Out-of-Distribution (OOD)** guardrails.

### 🧠 REFINED LOGIC (THE MATH)
To achieve >82% accuracy, the model logic was re-engineered to balance social and physical variables:
* **[LOG_SCALING]:** Wealth (Fare) was normalized using Log-scaling to prevent $500 outliers from hijacking the gradient.
* **[DECAY_PROTOCOL]:** Implemented Exponential Decay for Age > 80. Survival probability evaporates as data becomes scarce, preventing "cliff" errors.
* **[INTERACTION_LOGIC]:** Linked Pclass and Age to create a "Senior Class Penalty," reflecting historical casualty patterns.

### 🛡 DEFENSIVE PATTERNS
Built for the "Dirty Data" of a busy shift:
1.  **LEAKAGE_PURGE:** Explicitly stripped Ticket and Cabin fragments to prevent the model from "memorizing" individuals.
2.  **INPUT_SANITIZATION:** Automatic normalization of titles and strings (Mr, mR, MR -> Mr).
3.  **STOCHASTIC_TUNING:** Utilized 5-fold GridSearchCV to optimize XGBoost hyperparameters without overfitting.

---

## 🚀 DEPLOYMENT_LOGS

### INSTALL_DEPENDS
```bash
pip install xgboost pandas streamlit scikit-learn joblib

EXECUTE_TRAIN
Bash

# Calibrates XGBoost and generates titanic_model.pkl
python3 src/train_model.py

START_INTERFACE
Bash

# Launches the tactical dashboard
streamlit run src/app.py

🎙 PHILOSOPHY

    "In a bar or a codebase, you don't trust the data at face value. You build for the exceptions, not the rules. This project is a Digital Bouncer designed to handle outliers with precision."

👤 DEVELOPER_INFO

    Lead Engineer: Gary Edward Gaines, Jr.

    Focus: ML Ops, Defensive Engineering, NLP

    Location: Philadelphia, PA / Southern NJ Area

    Host_Machine: rosencrantz