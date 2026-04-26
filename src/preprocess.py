import pandas as pd
import numpy as np
import re

def process_titanic_titles(df):
    """
    Extracts titles and groups them into categories to reduce noise.
    """
    # Defensive check for 'name' vs 'Name'
    name_col = next((col for col in df.columns if col.lower() == 'name'), None)
    if name_col is None:
        raise KeyError("Target column 'name' (or 'Name') not found.")

    # Regex to grab the title
    df['Title'] = df[name_col].str.extract(' ([A-Za-z]+)\.', expand=False)

    # Mapping 'Top-Shelf' and rare titles
    title_mapping = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Special", "Countess": "Special", "Capt": "Special",
        "Col": "Special", "Don": "Special", "Dr": "Special",
        "Major": "Special", "Rev": "Special", "Sir": "Special",
        "Jonkheer": "Special", "Dona": "Special"
    }
    df['Title'] = df['Title'].replace(title_mapping)

    # Default unmapped titles to 'Special'
    standard_titles = ['Mr', 'Miss', 'Mrs', 'Master', 'Special']
    df.loc[~df['Title'].isin(standard_titles), 'Title'] = 'Special'
    df['Title'] = df['Title'].fillna('Special')

    return df

def refine_age_and_fare(df):
    """
    Imputes missing ages based on Title and applies log scaling to Fare.
    """
    # 1. Age Imputation: Use the median age of the specific Title group
    if 'Age' in df.columns:
        df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
        # Final fallback for any group that might have 0 age data
        df['Age'] = df['Age'].fillna(df['Age'].median())

    # 2. Fare Scaling: Squash outliers using Log Transformation
    if 'Fare' in df.columns:
        # Fill missing fare (common in test sets) with median
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        # Log(1+x) to handle zero values gracefully
        df['Fare'] = df['Fare'].apply(lambda x: np.log1p(x) if x > 0 else 0)
    
    return df

def encode_categorical_features(df, columns=['Title']):
    """
    Converts text categories into binary columns (One-Hot Encoding).
    """
    valid_cols = [col for col in columns if col in df.columns]
    # drop_first=True prevents the Dummy Variable Trap
    df = pd.get_dummies(df, columns=valid_cols, prefix=valid_cols, drop_first=True)
    return df

def run_full_preprocessing(df):
    """
    Executive function to run the full pipeline in order.
    """
    df = df.copy() # Protect original data
    
    df = process_titanic_titles(df)
    df = refine_age_and_fare(df)
    df = encode_categorical_features(df, columns=['Title'])
    
    # Drop high-cardinality or unneeded columns for the model
    cols_to_drop = ['PassengerId', 'name', 'Ticket', 'Cabin', 'Name']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df

if __name__ == "__main__":
    # Quick sanity check
    test_data = pd.DataFrame({'name': ['Gaines, Mr. Gary', 'Laina, Miss. Heikkinen'], 'Age': [30, np.nan], 'Fare': [7.25, 71.28]})
    print(run_full_preprocessing(test_data))