# src/feature_engineering.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """
    Load the dataset into a pandas DataFrame.
    """
    df = pd.read_csv(filepath)
    return df

def handle_missing_values(df):
    """
    Handle missing values appropriately.
    """

    df['credit_score'].fillna(df['credit_score'].median(),inplace = True)
    df['age'].fillna(df['age'].median(),inplace = True)
    df['tenure'].fillna(df['tenure'].median(),inplace = True)
    df['estimated_salary'].fillna(df['estimated_salary'].median(),inplace = True)
    df['balance'].fillna(0,inplace = True)
    df['products_number'].fillna(df['products_number'].mode(),inplace = True)
    df['gender'].fillna(df['gender'].mode(),inplace = True)
    df['credit_card'].fillna(df['credit_card'].mode(),inplace = True)
    df['active_member'].fillna(df['active_member'].mode(),inplace = True)

    return df

def create_age_groups(df):
    bins = [17, 30, 45, 60, 100]  # Age ranges
    labels = ['young', 'adult', 'middle-aged', 'senior']  # Labels for age groups
    #todo: create age groups
    df['age_group'] = pd.cut(df['age'], bins=bins, labels = labels, right = False)
    df = df.dropna()
    return df

def encode_categorical_features(df):
    """
    Encode categorical features using Label Encoding.
    """
    le = LabelEncoder()
    categorical_cols = ['country', 'gender', 'credit_card','age_group']

    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    return df

def save_processed_data(df, filepath):
    """
    Save the processed DataFrame to a CSV file.
    """
    df.to_csv(filepath,index = False)
    #todo save the processed data into data folder using to_csv

def main():
    # Load data
    df = load_data("/workspaces/ts-aiml-datascience-challenge-naveengowdaj03/data/bank_churn.csv")

    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Create age groups
    df = create_age_groups(df)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Save processed data
    save_processed_data(df, '/workspaces/ts-aiml-datascience-challenge-naveengowdaj03/data/processed_bank_churn.csv') 

if __name__ == "__main__":
    main()
