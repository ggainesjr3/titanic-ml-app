import pandas as pd
import numpy as np

def run_full_preprocessing(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # 1. Feature Engineering: Titles
    name_col = next((col for col in df.columns if 'name' in col), None)
    if name_col:
        df['title'] = df[name_col].str.extract(' ([A-Za-z]+)\.', expand=False)
        mapping = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Special", 
                   "Countess": "Special", "Capt": "Special", "Col": "Special", 
                   "Don": "Special", "Dr": "Special", "Major": "Special", 
                   "Rev": "Special", "Sir": "Special", "Jonkheer": "Special", "Dona": "Special"}
        df['title'] = df['title'].replace(mapping).fillna('Special')

    # 2. DEFENSIVE AGE CLIPPING
    # No one on the Titanic was 99. We clip at 80 to keep the model in reality.
    if 'age' in df.columns:
        df['age'] = df['age'].clip(lower=0.42, upper=80) 
        df['age'] = df.groupby('title')['age'].transform(lambda x: x.fillna(x.median()))
        
        # New Feature: Is the passenger in a high-risk age group?
        df['is_vulnerable'] = ((df['age'] < 5) | (df['age'] > 75)).astype(int)

    if 'fare' in df.columns:
        df['fare'] = df['fare'].fillna(df['fare'].median())
        df['fare'] = df['fare'].apply(lambda x: np.log1p(x) if x > 0 else 0)

    # 3. Encoding
    cols_to_encode = ['title', 'sex', 'embarked']
    present_cols = [c for c in cols_to_encode if c in df.columns]
    df = pd.get_dummies(df, columns=present_cols, drop_first=True)

    drop_list = ['passengerid', 'name', 'ticket', 'cabin']
    df = df.drop(columns=[c for c in drop_list if c in df.columns])
    
    return df.fillna(0)
