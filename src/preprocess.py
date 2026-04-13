import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    # 1. FEATURE ENGINEERING (Add your new logic here)
    # Extract Title before dropping the Name column
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Standardize titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                       'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Create FamilySize first so we can use it for IsAlone
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Create IsAlone feature
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    # 2. DATA CLEANING
    # Drop columns that are no longer needed
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median()) # Good practice for test sets
    
    # 3. ENCODING CATEGORICAL DATA
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    
    # Convert 'Title' and 'Embarked' into numeric columns (One-Hot Encoding)
    df = pd.get_dummies(df, columns=['Title', 'Embarked'])
    
    return df
