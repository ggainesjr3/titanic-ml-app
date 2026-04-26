import pandas as pd
import joblib
import os

# Assume src is in the Python path, allowing direct import
from src.preprocess import clean_data

def predict_test_set(model_path='models/titanic_model.pkl', test_data_path='data/test.csv', output_path='predictions.csv'):
    """
    Generates predictions on the test dataset and saves them to a CSV file.
    """
    # 1. Load Model and Features
    try:
        model_artifact = joblib.load(model_path)
        model = model_artifact['model']
        model_features = model_artifact['features']
        print(f"Model and features loaded from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please ensure the model has been trained and saved.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load Test Data
    try:
        df_test = pd.read_csv(test_data_path)
        print(f"Test data loaded from {test_data_path}")
    except FileNotFoundError:
        print(f"Error: Test data file not found at {test_data_path}. Please ensure test.csv is in the data directory.")
        return
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # Store PassengerId for submission file
    # Make sure 'PassengerId' exists in the test data, as it's crucial for submission
    if 'PassengerId' not in df_test.columns:
        print("Error: 'PassengerId' column not found in test data. Cannot create submission file.")
        return
    test_passenger_ids = df_test['PassengerId']

    # 3. Preprocess Test Data
    try:
        # It's good practice to work on a copy to avoid modifying the original DataFrame
        df_test_processed = clean_data(df_test.copy())
        print("Test data preprocessed.")
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return

    # 4. Reindex to match model features
    # Ensure the test data columns match the training data columns expected by the model
    try:
        df_test_processed = df_test_processed.reindex(columns=model_features, fill_value=0)
        print("Test data columns reindexed to match model features.")
    except Exception as e:
        print(f"Error reindexing test data columns: {e}. Ensure clean_data produces columns compatible with model features.")
        return
        
    # 5. Make Predictions
    try:
        predictions = model.predict(df_test_processed)
        print(f"Predictions generated for {len(predictions)} passengers.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # 6. Create Submission File (Kaggle format)
    try:
        submission_df = pd.DataFrame({
            'PassengerId': test_passenger_ids,
            'Survived': predictions
        })
        submission_df.to_csv(output_path, index=False)
        print(f"Submission file created at {output_path}")
    except Exception as e:
        print(f"Error creating submission file: {e}")
        return

if __name__ == "__main__":
    # This script is intended to be run from the project's root directory.
    # The PYTHONPATH environment variable is typically set up by the CLI,
    # allowing 'from src.preprocess import clean_data' to work.
    
    # Check if data and models directories exist
    if not os.path.exists('data') or not os.path.exists('models'):
        print("Error: 'data' or 'models' directory not found. Please ensure they exist.")
    elif not os.path.exists('data/test.csv'):
        print("Error: 'data/test.csv' not found. Please ensure it has been downloaded.")
    elif not os.path.exists('models/titanic_model.pkl'):
        print("Error: 'models/titanic_model.pkl' not found. Please ensure the model has been trained.")
    else:
        predict_test_set()
