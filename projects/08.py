"""
Toy Sales Regression Analysis
==============================
This script performs Simple and Multiple Linear Regression analysis on toy sales data.

Original: R notebook (08.ipynb)
Converted to: Python script
Date: October 3, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import argparse
import sys


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
        # Clean column names (remove trailing/leading spaces)
        df.columns = df.columns.str.strip()
        
        print("\n" + "="*60)
        print("TOY SALES REGRESSION ANALYSIS")
        print("="*60)
        print("Python implementation of R notebook analysis")
        print("="*60)
        print("Data loaded successfully!")
        print(f"\nFirst few rows of the dataset:")
        print(df.head())
        print(f"\nDataset shape: {df.shape}")
        print(f"\nColumn names: {df.columns.tolist()}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found!")
        print("Please ensure the dataset exists in the correct location.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


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
        print("Decision: Fail to reject H0 - No significant relationship")
    
    print("\nPredicted values (first 5):")
    print(predictions[:5])
    
    print("\nResiduals/Errors (first 5):")
    print(errors[:5])
    
    return model, predictions, errors, (intercept, slope), r_squared, p_value


def plot_simple_regression(df, predictions, save_path='simple_regression_plot.png'):
    """
    Create scatter plot with regression line for Simple Linear Regression.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing Price and Unitsales columns
    predictions : np.array
        Predicted values from the regression model
    save_path : str
        Path to save the plot
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATION")
    print("="*60)
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(df['Price'], df['Unitsales'], alpha=0.6, s=100, 
                color='steelblue', edgecolors='black', linewidth=0.5,
                label='Actual Data')
    
    # Regression line
    plt.plot(df['Price'], predictions, color='red', linewidth=2, 
             label='Regression Line')
    
    plt.xlabel('Price ($)', fontsize=12, fontweight='bold')
    plt.ylabel('Unit Sales', fontsize=12, fontweight='bold')
    plt.title('Simple Linear Regression: Unit Sales vs Price', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add equation to plot
    model_temp = LinearRegression()
    model_temp.fit(df[['Price']].values, df['Unitsales'].values)
    equation = f'Unitsales = {model_temp.intercept_:.2f} + ({model_temp.coef_[0]:.2f}) × Price'
    plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def multiple_linear_regression(df):
    """
    Perform Multiple Linear Regression: Unitsales ~ Price + Adexp + Promexp
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing all predictor columns
        
    Returns:
    --------
    tuple
        (model, predictions, r_squared, coefficients_info)
    """
    print("\n" + "="*60)
    print("MULTIPLE LINEAR REGRESSION")
    print("Unitsales ~ Price + Adexp + Promexp")
    print("="*60)
    
    # Prepare data
    X = df[['Price', 'Adexp', 'Promexp']].values
    y = df['Unitsales'].values
    feature_names = ['Price', 'Adexp', 'Promexp']
    
    # Fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get predictions
    predictions = model.predict(X)
    
    # Calculate errors
    errors = y - predictions
    
    # Get coefficients
    intercept = model.intercept_
    coefficients = model.coef_
    
    # Calculate R-squared
    r_squared = model.score(X, y)
    
    # Calculate statistics for each coefficient
    n = len(y)
    k = len(coefficients)  # number of predictors
    dof = n - k - 1  # degrees of freedom
    
    # Residual standard error
    residual_std = np.sqrt(np.sum(errors**2) / dof)
    
    # Standard errors and p-values for each coefficient
    X_with_intercept = np.column_stack([np.ones(n), X])
    var_coef = residual_std**2 * np.linalg.inv(X_with_intercept.T @ X_with_intercept).diagonal()
    se_coefs = np.sqrt(var_coef)
    
    t_stats = np.concatenate([[intercept], coefficients]) / se_coefs
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
    
    # Print results
    print("\nRegression Equation:")
    equation = f"Unitsales = {intercept:.2f}"
    for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
        equation += f" + ({coef:.2f}) * {name}"
    print(equation)
    
    print(f"\nIntercept: {intercept:.2f}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Adjusted R-squared: {1 - (1 - r_squared) * (n - 1) / dof:.4f}")
    
    print("\n--- Coefficient Summary ---")
    print(f"{'Variable':<15} {'Coefficient':<15} {'Std Error':<15} {'t-stat':<12} {'p-value':<12}")
    print("-" * 70)
    print(f"{'Intercept':<15} {intercept:<15.2f} {se_coefs[0]:<15.4f} {t_stats[0]:<12.4f} {p_values[0]:<12.6f}")
    
    for i, name in enumerate(feature_names):
        print(f"{name:<15} {coefficients[i]:<15.2f} {se_coefs[i+1]:<15.4f} {t_stats[i+1]:<12.4f} {p_values[i+1]:<12.6f}")
    
    # Variable importance (based on p-values)
    print("\n--- Variable Importance (by significance) ---")
    coef_data = list(zip(feature_names, coefficients, p_values[1:]))
    coef_data_sorted = sorted(coef_data, key=lambda x: x[2])
    
    for i, (name, coef, p_val) in enumerate(coef_data_sorted, 1):
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"{i}. {name}: p-value = {p_val:.6f} {significance}")
    
    print("\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    
    # Interpretation
    print("\n--- Interpretation ---")
    alpha = 0.05
    significant_vars = [name for name, _, p_val in coef_data if p_val < alpha]
    
    if significant_vars:
        print(f"Significant predictors (α = {alpha}): {', '.join(significant_vars)}")
    else:
        print(f"No significant predictors at α = {alpha}")
    
    coefficients_info = {
        'names': feature_names,
        'values': coefficients,
        'p_values': p_values[1:],
        'intercept': intercept
    }
    
    return model, predictions, r_squared, coefficients_info


def predict_scenarios(model):
    """
    Predict unit sales for two business scenarios.
    
    Parameters:
    -----------
    model : LinearRegression
        Fitted multiple linear regression model
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with scenario predictions
    """
    print("\n" + "="*60)
    print("SCENARIO PREDICTIONS")
    print("="*60)
    
    # Define scenarios
    scenarios = pd.DataFrame({
        'Price': [9.1, 8.1],
        'Adexp': [52, 50],
        'Promexp': [61, 60]
    })
    
    # Make predictions
    predictions = model.predict(scenarios)
    
    # Display results
    print("\nScenario 1:")
    print(f"  Price: ${scenarios.loc[0, 'Price']}")
    print(f"  Advertising Expenditure: ${scenarios.loc[0, 'Adexp']}k")
    print(f"  Promotion Expenditure: ${scenarios.loc[0, 'Promexp']}k")
    print(f"  Predicted Unit Sales: {predictions[0]:,.0f}")
    
    print("\nScenario 2:")
    print(f"  Price: ${scenarios.loc[1, 'Price']}")
    print(f"  Advertising Expenditure: ${scenarios.loc[1, 'Adexp']}k")
    print(f"  Promotion Expenditure: ${scenarios.loc[1, 'Promexp']}k")
    print(f"  Predicted Unit Sales: {predictions[1]:,.0f}")
    
    # Comparison
    difference = predictions[1] - predictions[0]
    print("\n--- Comparison ---")
    print(f"Difference in predicted sales: {difference:,.0f} units")
    
    if difference > 0:
        percent_increase = (difference / predictions[0]) * 100
        print(f"Scenario 2 is better by {difference:,.0f} units ({percent_increase:.2f}% increase)")
        print("\n✓ RECOMMENDATION: Choose Scenario 2")
        print(f"  - Lower price (${scenarios.loc[1, 'Price']} vs ${scenarios.loc[0, 'Price']})")
        print(f"  - Lower marketing costs")
        print(f"  - Higher projected sales")
    else:
        percent_increase = (abs(difference) / predictions[1]) * 100
        print(f"Scenario 1 is better by {abs(difference):,.0f} units ({percent_increase:.2f}% increase)")
        print("\n✓ RECOMMENDATION: Choose Scenario 1")
    
    scenarios['Predicted_Sales'] = predictions
    return scenarios


def main():
    """
    Main function to orchestrate the analysis workflow.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Toy Sales Regression Analysis - Convert R notebook to Python'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='../datasets/Toy_sales_csv.csv',
        help='Path to the toy sales CSV file'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip generating plots'
    )
    
    args = parser.parse_args()
    
    # Set style for plots
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # Step 1: Load data
    df = load_data(args.data)
    
    # Step 2: Simple Linear Regression
    slr_model, slr_predictions, slr_errors, slr_coefs, slr_r2, slr_pval = simple_linear_regression(df)
    
    # Step 3: Visualization (if not disabled)
    if not args.no_plot:
        plot_simple_regression(df, slr_predictions, '../datasets/simple_regression_plot.png')
    
    # Step 4: Multiple Linear Regression
    mlr_model, mlr_predictions, mlr_r2, mlr_coefs = multiple_linear_regression(df)
    
    # Step 5: Scenario Predictions
    scenario_results = predict_scenarios(mlr_model)
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nSimple Linear Regression R²: {slr_r2:.4f}")
    print(f"Multiple Linear Regression R²: {mlr_r2:.4f}")
    print(f"Improvement: {(mlr_r2 - slr_r2):.4f} ({((mlr_r2 - slr_r2)/slr_r2)*100:.2f}%)")
    print("\nAll analyses completed successfully! ✓")
    print("="*60)


if __name__ == "__main__":
    main()
