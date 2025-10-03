"""
Toy Sales Regression Analysis
==============================
This script performs Simple and Multiple Linear Regression analysis on toy sales data.

Original: R notebook (08.ipynb)
Converted to : Python 
Date: October 3, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import argparse
import os


def load_data(filepath):
    """
    Load the toy sales dataset from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing toy sales data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the toy sales data
    """
    try:
        df = pd.read_csv(filepath)
        # Clean column names by removing trailing/leading spaces
        df.columns = df.columns.str.strip()
        print("Data loaded successfully!")
        print("\nFirst few rows of the dataset:")
        print(df.head())
        print("\nDataset shape:", df.shape)
        print("\nColumn names:", df.columns.tolist())
        return df
    except FileNotFoundError:
        print(f"Error: File {filepath} not found!")
        return None


def simple_linear_regression(df):
    """
    Perform Simple Linear Regression: Unitsales ~ Price
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing Price and Unitsales columns
        
    Returns:
    --------
    tuple
        (model, predictions, errors, coefficients, r_squared, p_value)
    """
    print("\n" + "="*60)
    print("SIMPLE LINEAR REGRESSION: Unitsales ~ Price")
    print("="*60)
    
    # Prepare data
    X = df[['Price']].values
    y = df['Unitsales'].values
    
    # Fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get predictions
    predictions = model.predict(X)
    
    # Calculate errors (residuals)
    errors = y - predictions
    
    # Get coefficients
    intercept = model.intercept_
    slope = model.coef_[0]
    
    # Calculate R-squared
    r_squared = model.score(X, y)
    
    # Calculate p-value for the slope
    # Using scipy.stats for statistical significance
    n = len(y)
    dof = n - 2  # degrees of freedom
    residual_std = np.sqrt(np.sum(errors**2) / dof)
    se_slope = residual_std / np.sqrt(np.sum((X - X.mean())**2))
    t_stat = slope / se_slope
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), dof))
    
    # Print results
    print("\nRegression Equation:")
    print(f"Unitsales = {intercept:.2f} + ({slope:.2f}) * Price")
    print(f"\nIntercept (b0): {intercept:.2f}")
    print(f"Slope (b1): {slope:.2f}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"P-value for slope: {p_value:.6f}")
    
    # Hypothesis test interpretation
    print("\n--- Hypothesis Test ---")
    print("H0: β1 = 0 (No relationship between Price and Unitsales)")
    print("Ha: β1 ≠ 0 (Relationship exists)")
    alpha = 0.05
    if p_value < alpha:
        print(f"Result: p-value ({p_value:.6f}) < α ({alpha})")
        print("Decision: Reject H0 - Price significantly affects Unitsales")
    else:
        print(f"Result: p-value ({p_value:.6f}) >= α ({alpha})")
        print("Decision: Fail to reject H0 - No significant effect")
    
    print("\nPredicted values (first 5):")
    print(predictions[:5])
    
    print("\nResiduals/Errors (first 5):")
    print(errors[:5])
    
    return model, predictions, errors, (intercept, slope), r_squared, p_value


def plot_simple_regression(df, predictions):
    """
    Create scatter plot with regression line for Simple Linear Regression.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing Price and Unitsales columns
    predictions : np.array
        Predicted values from the regression model
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATION")
    print("="*60)
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(df['Price'], df['Unitsales'], alpha=0.6, s=100, 
                edgecolors='black', label='Actual Data')
    
    # Regression line
    plt.plot(df['Price'], predictions, color='red', linewidth=2, 
             label='Regression Line')
    
    plt.xlabel('Price ($)', fontsize=12)
    plt.ylabel('Unit Sales', fontsize=12)
    plt.title('Simple Linear Regression: Unit Sales vs Price', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = '../datasets/simple_regression_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    plt.show()


def multiple_linear_regression(df):
    """
    Perform Multiple Linear Regression: Unitsales ~ Price + Adexp + Promexp
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing all predictor and response variables
        
    Returns:
    --------
    LinearRegression
        Fitted multiple regression model
    """
    print("\n" + "="*60)
    print("MULTIPLE LINEAR REGRESSION: Unitsales ~ Price + Adexp + Promexp")
    print("="*60)
    
    # Prepare data
    X = df[['Price', 'Adexp', 'Promexp']].values
    y = df['Unitsales'].values
    
    # Fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get coefficients
    intercept = model.intercept_
    coefficients = model.coef_
    
    # Calculate R-squared
    r_squared = model.score(X, y)
    
    # Calculate p-values for each coefficient
    predictions = model.predict(X)
    residuals = y - predictions
    n = len(y)
    p = X.shape[1]  # number of predictors
    dof = n - p - 1  # degrees of freedom
    
    residual_std = np.sqrt(np.sum(residuals**2) / dof)
    
    # Calculate standard errors and p-values
    X_with_intercept = np.column_stack([np.ones(n), X])
    var_covar_matrix = residual_std**2 * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    se = np.sqrt(np.diag(var_covar_matrix))
    
    t_stats = np.concatenate([[intercept], coefficients]) / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
    
    # Print results
    print("\nRegression Equation:")
    print(f"Unitsales = {intercept:.2f} + ({coefficients[0]:.2f}) * Price + ({coefficients[1]:.2f}) * Adexp + ({coefficients[2]:.2f}) * Promexp")
    
    print("\n--- Coefficients and Significance ---")
    predictors = ['Intercept', 'Price', 'Adexp', 'Promexp']
    coefs = [intercept] + list(coefficients)
    
    print(f"\n{'Variable':<15} {'Coefficient':<15} {'Std Error':<15} {'t-stat':<15} {'p-value':<15}")
    print("-" * 75)
    for i, var in enumerate(predictors):
        print(f"{var:<15} {coefs[i]:<15.4f} {se[i]:<15.4f} {t_stats[i]:<15.4f} {p_values[i]:<15.6f}")
    
    print(f"\nR-squared: {r_squared:.4f}")
    
    # Interpretation
    print("\n--- Variable Importance (based on p-values) ---")
    sorted_indices = np.argsort(p_values[1:])  # Exclude intercept
    for idx in sorted_indices:
        var_name = predictors[idx + 1]
        p_val = p_values[idx + 1]
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"{var_name}: p-value = {p_val:.6f} {significance}")
    
    alpha = 0.05
    print(f"\nNote: Variables with p-value < {alpha} are statistically significant")
    print("Most important variable (lowest p-value): " + predictors[1:][sorted_indices[0]])
    
    return model


def predict_scenarios(model):
    """
    Predict unit sales for two different pricing and marketing scenarios.
    
    Parameters:
    -----------
    model : LinearRegression
        Fitted multiple regression model
    """
    print("\n" + "="*60)
    print("PREDICTION SCENARIOS")
    print("="*60)
    
    # Define scenarios
    scenarios = pd.DataFrame({
        'Price': [9.1, 8.1],
        'Adexp': [52.0, 50.0],
        'Promexp': [61.0, 60.0]
    })
    
    print("\nScenario 1: Price=$9.1, Adexp=$52,000, Promexp=$61,000")
    print("Scenario 2: Price=$8.1, Adexp=$50,000, Promexp=$60,000")
    
    # Make predictions
    predictions = model.predict(scenarios)
    
    print("\n--- Predicted Unit Sales ---")
    for i, pred in enumerate(predictions, 1):
        print(f"Scenario {i}: {pred:,.2f} units")
    
    # Determine best scenario
    best_scenario = np.argmax(predictions) + 1
    max_sales = np.max(predictions)
    
    print("\n--- CONCLUSION ---")
    print(f"Scenario {best_scenario} is expected to yield the maximum unit sales:")
    print(f"Maximum Unit Sales: {max_sales:,.2f} units")
    
    if best_scenario == 1:
        print("\nRecommendation: Choose Scenario 1")
        print("Despite higher price, the increased advertising and promotion budget")
        print("is expected to generate more unit sales.")
    else:
        print("\nRecommendation: Choose Scenario 2")
        print("The lower price point combined with marketing expenditure")
        print("is expected to generate more unit sales.")


def main():
    """
    Main function to orchestrate the entire analysis pipeline.
    """
    # Set up argument parser for command-line usage
    parser = argparse.ArgumentParser(
        description='Toy Sales Regression Analysis - Python Version'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='../datasets/Toy_sales_csv.csv',
        help='Path to the toy sales CSV file (default: ../datasets/Toy_sales_csv.csv)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plotting visualizations'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("="*60)
    print("TOY SALES REGRESSION ANALYSIS")
    print("="*60)
    print("Python implementation of R notebook analysis")
    print("="*60)
    
    # Load data
    df = load_data(args.data)
    if df is None:
        return
    
    # Simple Linear Regression
    slr_model, slr_predictions, slr_errors, slr_coefs, slr_r2, slr_pval = simple_linear_regression(df)
    
    # Plot Simple Linear Regression (unless --no-plot flag is used)
    if not args.no_plot:
        plot_simple_regression(df, slr_predictions)
    
    # Multiple Linear Regression
    mlr_model = multiple_linear_regression(df)
    
    # Prediction Scenarios
    predict_scenarios(mlr_model)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    # Set plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # Run main analysis
    main()
