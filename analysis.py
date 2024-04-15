import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filepath):
    """
    Load the combined data from a CSV file.
    """
    return pd.read_csv(filepath, parse_dates=['SETTLEMENTDATE'])

def clean_data(df):
    """
    Perform initial data cleaning steps:
    - Check for missing values and handle them (e.g., imputation or removal).
    - Verify data types are correct (especially datetime and numeric types).
    """
    # Example: Impute missing values with the mean of the column
    df.fillna(df.mean(), inplace=True)
    return df

def exploratory_data_analysis(df):
    """
    Conduct exploratory data analysis:
    - Summary statistics
    - Distribution of key variables
    - Correlation analysis
    """
    print(df.describe())
    # Histograms for numeric data
    df.hist(figsize=(15, 10))
    plt.show()
    # Pairplot for seeing pairwise relationships
    sns.pairplot(df.select_dtypes(include=[np.number]))
    plt.show()
    # Heatmap for correlation matrix
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

def hypothesis_testing(df):
    """
    Perform hypothesis testing:
    - Example: Test if the mean RRP is significantly different between two regions.
    """
    region1 = df['DP_RRP_NSW1']
    region2 = df['DP_RRP_QLD1']
    t_stat, p_val = ttest_ind(region1, region2)
    print(f"T-test result: T-stat={t_stat}, P-value={p_val}")

def regression_analysis(df):
    """
    Conduct regression analysis to predict RRP based on other variables:
    - Simple linear regression as an example.
    """
    # Select a dependent variable and one independent variable for simplicity
    X = df[['DRS_TOTALDEMAND_NSW1']]  # Independent variable
    y = df['DP_RRP_NSW1']  # Dependent variable

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}, R^2 Score: {r2}")

def main():
    """
    Main function to run the analysis steps.
    """
    filepath = 'combined_5min_data_head.csv'
    df = load_data(filepath)
    df = clean_data(df)
    exploratory_data_analysis(df)
    hypothesis_testing(df)
    regression_analysis(df)

if __name__ == "__main__":
    main()