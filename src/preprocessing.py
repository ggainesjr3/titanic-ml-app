import pandas as pd
import numpy as np

def run_full_preprocessing(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # 1. Title Extraction (Safe Predictor)
    if 'name' in df.columns:
        df['title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        mapping = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Special", 
                   "Countess": "Special", "Capt": "Special", "Col": "Special", 
                   "Don": "Special", "Dr": "Special", "Major": "Special", 
                   "Rev": "Special", "Sir": "Special", "Jonkheer": "Special", "Dona": "Special"}
        df['title'] = df['title'].replace(mapping).fillna('Special')

    # 2. Age (Capped to 80 for realism)
    if 'age' in df.columns:
        df['age'] = df['age'].clip(upper=80)
        df['age'] = df['age'].fillna(df['age'].median())
        
    # 3. Fare (Log-scaled to remove outlier 'shield' effect)
    if 'fare' in df.columns:
        df['fare'] = df['fare'].fillna(df['fare'].median())
        df['fare_log'] = np.log1p(df['fare'])

    # 4. Social Context
    if 'sibsp' in df.columns and 'parch' in df.columns:
        df['family_size'] = df['sibsp'] + df['parch'] + 1

    # 5. Encoding
    cols_to_encode = ['title', 'sex', 'embarked']
    present_cols = [c for c in cols_to_encode if c in df.columns]
    df = pd.get_dummies(df, columns=present_cols, drop_first=True)

    # 6. THE PURGE: Removing leakage magnets
    drop_list = ['passengerid', 'name', 'ticket', 'cabin', 'fare', 'sibsp', 'parch', 'survived']
    df = df.drop(columns=[c for c in drop_list if c in df.columns])
    
    return df.fillna(0)
