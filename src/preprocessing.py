import pandas as pd
import numpy as np

def run_full_preprocessing(df):
    df = df.copy()
    # Force lowercase for defensive engineering
    df.columns = [c.lower() for c in df.columns]
    
    # Extract Titles
    name_col = next((col for col in df.columns if 'name' in col), None)
    if name_col:
        df['title'] = df[name_col].str.extract(' ([A-Za-z]+)\.', expand=False)
        mapping = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Special", 
                   "Countess": "Special", "Capt": "Special", "Col": "Special", 
                   "Don": "Special", "Dr": "Special", "Major": "Special", 
                   "Rev": "Special", "Sir": "Special", "Jonkheer": "Special", "Dona": "Special"}
        df['title'] = df['title'].replace(mapping).fillna('Special')

    # Impute Age based on title median
    if 'age' in df.columns:
        df['age'] = df.groupby('title')['age'].transform(lambda x: x.fillna(x.median()))
        df['age'] = df['age'].fillna(df['age'].median())

    # Handle Fare outliers
    if 'fare' in df.columns:
        df['fare'] = df['fare'].fillna(df['fare'].median())
        df['fare'] = df['fare'].apply(lambda x: np.log1p(x) if x > 0 else 0)

    # One-Hot Encoding (Fixes the 'female' string error)
    cols_to_encode = ['title', 'sex', 'embarked']
    present_cols = [c for c in cols_to_encode if c in df.columns]
    df = pd.get_dummies(df, columns=present_cols, drop_first=True)

    # Drop non-numeric columns
    drop_list = ['passengerid', 'name', 'ticket', 'cabin']
    df = df.drop(columns=[c for c in drop_list if c in df.columns])
    
    return df.fillna(0)
