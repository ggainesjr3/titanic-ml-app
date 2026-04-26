import pandas as pd
import numpy as np

def run_full_preprocessing(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # 1. Title Extraction
    name_col = next((col for col in df.columns if 'name' in col), None)
    if name_col:
        df['title'] = df[name_col].str.extract(' ([A-Za-z]+)\.', expand=False)
        mapping = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Special", 
                   "Countess": "Special", "Capt": "Special", "Col": "Special", 
                   "Don": "Special", "Dr": "Special", "Major": "Special", 
                   "Rev": "Special", "Sir": "Special", "Jonkheer": "Special", "Dona": "Special"}
        df['title'] = df['title'].replace(mapping).fillna('Special')

    # 2. Advanced Age & Fare Handling
    if 'age' in df.columns:
        df['age'] = df['age'].clip(upper=80)
        df['age'] = df.groupby('title')['age'].transform(lambda x: x.fillna(x.median()))
        
    if 'fare' in df.columns:
        df['fare'] = df['fare'].fillna(df['fare'].median())
        # Log scaling prevents "Super Rich" outliers from breaking the model
        df['fare_log'] = np.log1p(df['fare'])

    # 3. INTERACTION TERMS (The Balance Fix)
    if 'age' in df.columns and 'pclass' in df.columns:
        # High value = High risk (Old + Lower Class)
        df['age_class_risk'] = df['age'] * df['pclass']
        # Survival penalty for outliers
        df['senior_pclass_penalty'] = ((df['age'] > 60) & (df['pclass'] > 1)).astype(int)

    # 4. Encoding & Cleanup
    cols_to_encode = ['title', 'sex', 'embarked']
    present_cols = [c for c in cols_to_encode if c in df.columns]
    df = pd.get_dummies(df, columns=present_cols, drop_first=True)

    drop_list = ['passengerid', 'name', 'ticket', 'cabin', 'fare'] # Drop raw fare, use log
    df = df.drop(columns=[c for c in drop_list if c in df.columns])
    
    return df.fillna(0)
