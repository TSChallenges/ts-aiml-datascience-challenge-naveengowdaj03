import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt         #  importing some basic libraries to work with the dataset and for visualization



def load_data(filepath):
    """
    Load the dataset into a pandas DataFrame.
    """
    df = pd.read_csv(filepath)
    return df

def check_missing_values(df):
    """
    Check for missing values in the DataFrame.
    """
    missing = df.isnull().sum()
    print("Missing Values:\n", missing)

def generate_summary_statistics(df):
    """
    Generate summary statistics for key variables.
    """
    summary =df.describe()
    print("Summary Statistics:\n", summary)

def visualize_distributions(df):
    """
    Visualize distributions of age, balance, credit_score, and estimated_salary.
    
    Hint: use sns.histplot() 
    """
    fig, axes = plt.subplots(2,2,figsize=(14,10))
    fig.suptitle("Distribution")

    sns.histplot(df['age'],kde = True , ax= axes[0,0], color = "skyblue")
    axes[0,0].set_title("age")


    sns.histplot(df['balance'],kde = True , ax= axes[0,1], color = "skyblue")
    axes[0,1].set_title("balance")

    sns.histplot(df['credit_score'],kde = True , ax= axes[1,0], color = "skyblue")
    axes[1,0].set_title("credit_score")

    sns.histplot(df['estimated_salary'],kde = True , ax= axes[1,1], color = "skyblue")
    axes[1,1].set_title("estimated_salary")
    
    plt.tight_layout(rect = [0,0.03,1,0.95])
    plt.show()

def main():
    # Load data
    df = load_data("/workspaces/ts-aiml-datascience-challenge-naveengowdaj03/data/bank_churn.csv")

    # Check for missing values
    check_missing_values(df)
    
    # Generate summary statistics
    generate_summary_statistics(df)
    
    # Visualize distributions
    visualize_distributions(df)

if __name__ == "__main__":
    main()