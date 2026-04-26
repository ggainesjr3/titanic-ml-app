import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load your full dataset
# Make sure titanic_full.csv is actually in this folder!
try:
    df = pd.read_csv('titanic_full.csv')

    # Split into 80% training and 20% testing
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save the test set
    test_df.to_csv('test_data.csv', index=False)
    print("Successfully created: test_data.csv")
    
except FileNotFoundError:
    print("Error: 'titanic_full.csv' not found in this directory.")
