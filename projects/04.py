#!/usr/bin/env python3
"""
Binomial Distribution Analysis: Travel Abroad Data

This script analyzes travel abroad data using binomial distribution theory.
It calculates probabilities for different scenarios and creates visualizations
to understand the statistical patterns.

Author: Data Science Analysis
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import os

# Set up plotting style for better visuals
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data(file_path):
    """
    Load the travel abroad dataset and perform initial exploration.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
        float: Percentage of people who traveled abroad
    """
    print("=" * 60)
    print("📊 LOADING AND EXPLORING TRAVEL DATA")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv(file_path)
    print(f"✅ Dataset loaded successfully!")
    print(f"📈 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Display first few rows
    print("\n🔍 First 5 rows of the dataset:")
    print(df.head())
    
    # Count people who traveled abroad
    traveled_count = len(df[df['Travelledabroad'] == 'Y'])
    not_traveled_count = len(df[df['Travelledabroad'] == 'N'])
    total_count = len(df)
    
    print(f"\n📊 Travel Statistics:")
    print(f"   👥 Total people surveyed: {total_count}")
    print(f"   ✈️  People who traveled abroad: {traveled_count}")
    print(f"   🏠 People who didn't travel abroad: {not_traveled_count}")
    
    # Calculate percentage
    percent_traveled = (traveled_count / total_count) * 100
    print(f"   📈 Percentage who traveled abroad: {percent_traveled:.2f}%")
    
    return df, percent_traveled

def calculate_binomial_probabilities_n10(p):
    """
    Calculate binomial probabilities for sample size n=10.
    
    This function calculates the probability of exactly k people having
    traveled abroad in a random sample of 10 people, for k from 0 to 10.
    
    Args:
        p (float): Probability of success (proportion who traveled abroad)
        
    Returns:
        dict: Dictionary containing all calculated probabilities
    """
    print("\n" + "=" * 60)
    print("🎲 BINOMIAL PROBABILITY CALCULATIONS (n=10)")
    print("=" * 60)
    
    n = 10  # Sample size
    probabilities = {}
    
    print(f"📋 Using binomial distribution with:")
    print(f"   • Sample size (n): {n}")
    print(f"   • Success probability (p): {p:.4f}")
    print(f"   • This satisfies n*p = {n*p:.2f} >= 10? {'Yes' if n*p >= 10 else 'No'}")
    
    # Calculate probabilities for each scenario
    scenarios = [
        ("no one", 0),
        ("exactly 1 person", 1),
        ("exactly 2 persons", 2),
        ("exactly 3 persons", 3),
        ("exactly 4 persons", 4),
        ("exactly 5 persons", 5),
        ("exactly 6 persons", 6),
        ("exactly 7 persons", 7),
        ("exactly 8 persons", 8),
        ("exactly 9 persons", 9),
        ("all 10 persons", 10)
    ]
    
    print(f"\n🎯 Probability calculations:")
    for description, k in scenarios:
        prob = stats.binom.pmf(k, n, p)
        probabilities[k] = prob
        print(f"   P({description} traveled abroad) = {prob:.6f}")
    
    return probabilities

def create_probability_visualization_n10(probabilities):
    """
    Create a bar plot showing probability distribution for n=10.
    
    Args:
        probabilities (dict): Dictionary of probabilities for each k value
    """
    print(f"\n📊 Creating visualization for n=10 scenario...")
    
    # Prepare data for plotting (excluding k=0 as mentioned in R code)
    k_values = list(range(1, 11))  # 1 to 10 persons
    prob_values = [probabilities[k] for k in k_values]
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    bars = plt.bar(k_values, prob_values, 
                   color='skyblue', 
                   edgecolor='black', 
                   linewidth=1.2,
                   alpha=0.8)
    
    # Customize the plot
    plt.title('Probability Distribution: Number of Persons Who Traveled Abroad (n=10)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Number of Persons Who Have Traveled Abroad', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for bar, prob in zip(bars, prob_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{prob:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Set x-axis ticks
    plt.xticks(k_values)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('/workspaces/data-science/projects/binomial_n10_plot.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Plot saved as 'binomial_n10_plot.png'")

def analyze_normal_approximation_n100(p):
    """
    Analyze the scenario with n=100 using normal approximation.
    
    Args:
        p (float): Probability of success
        
    Returns:
        tuple: Mean, standard deviation, and probability calculations
    """
    print("\n" + "=" * 60)
    print("📈 NORMAL APPROXIMATION ANALYSIS (n=100)")
    print("=" * 60)
    
    n = 100
    
    # Calculate parameters for normal approximation
    mu = n * p  # Mean
    sigma = np.sqrt(n * p * (1 - p))  # Standard deviation
    
    print(f"📋 Normal approximation parameters:")
    print(f"   • Sample size (n): {n}")
    print(f"   • Success probability (p): {p:.4f}")
    print(f"   • Mean (μ = n*p): {mu:.2f}")
    print(f"   • Standard deviation (σ = √(n*p*(1-p))): {sigma:.2f}")
    print(f"   • Rule check: n*p = {n*p:.1f} ≥ 10? {'✅ Yes' if n*p >= 10 else '❌ No'}")
    print(f"   • Rule check: n*(1-p) = {n*(1-p):.1f} ≥ 10? {'✅ Yes' if n*(1-p) >= 10 else '❌ No'}")
    
    # Calculate probability using normal approximation
    # P(X ≥ 59) using normal approximation with continuity correction
    prob_normal_approx = 1 - stats.norm.cdf(58.5, loc=mu, scale=sigma)
    
    # Calculate exact probability using binomial distribution
    prob_exact = 1 - stats.binom.cdf(58, n, p)
    
    print(f"\n🎯 Probability that at least 59 persons traveled abroad:")
    print(f"   • Using normal approximation: {prob_normal_approx:.8f}")
    print(f"   • Using exact binomial: {prob_exact:.8f}")
    print(f"   • Difference: {abs(prob_normal_approx - prob_exact):.8f}")
    
    return mu, sigma, prob_normal_approx, prob_exact

def create_comparison_visualization(p):
    """
    Create visualizations comparing binomial and normal distributions.
    
    Args:
        p (float): Probability of success
    """
    print(f"\n📊 Creating comparison visualizations...")
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Binomial distribution for n=10
    n1 = 10
    k_values_10 = np.arange(0, n1 + 1)
    prob_values_10 = [stats.binom.pmf(k, n1, p) for k in k_values_10]
    
    ax1.bar(k_values_10, prob_values_10, color='lightcoral', alpha=0.7, edgecolor='black')
    ax1.set_title(f'Binomial Distribution (n={n1})', fontweight='bold')
    ax1.set_xlabel('Number of Successes (k)')
    ax1.set_ylabel('Probability')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Binomial vs Normal for n=100
    n2 = 100
    k_values_100 = np.arange(0, n2 + 1)
    prob_values_100 = [stats.binom.pmf(k, n2, p) for k in k_values_100]
    
    # Normal approximation
    mu = n2 * p
    sigma = np.sqrt(n2 * p * (1 - p))
    x_normal = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    y_normal = stats.norm.pdf(x_normal, mu, sigma)
    
    # Plot binomial as bars (but thinner since there are many)
    ax2.bar(k_values_100, prob_values_100, width=0.8, color='skyblue', 
            alpha=0.6, label='Binomial Distribution', edgecolor='none')
    
    # Plot normal approximation as a curve
    ax2.plot(x_normal, y_normal, 'r-', linewidth=2, 
             label='Normal Approximation')
    
    ax2.set_title(f'Binomial vs Normal Approximation (n={n2})', fontweight='bold')
    ax2.set_xlabel('Number of Successes (k)')
    ax2.set_ylabel('Probability Density')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspaces/data-science/projects/binomial_comparison_plots.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Comparison plots saved as 'binomial_comparison_plots.png'")

def main():
    """
    Main function that orchestrates the entire analysis.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze travel abroad data using binomial distribution')
    parser.add_argument('--data-path', 
                       default='../datasets/travelled abroad_csv.csv',
                       help='Path to the CSV data file')
    parser.add_argument('--skip-plots', 
                       action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    print("🚀 TRAVEL ABROAD DATA ANALYSIS")
    print("Using Binomial Distribution and Normal Approximation")
    print("=" * 60)
    
    try:
        # Step 1: Load and explore the data
        df, percent_traveled = load_and_explore_data(args.data_path)
        p = percent_traveled / 100  # Convert percentage to probability
        
        # Step 2: Calculate binomial probabilities for n=10
        probabilities_n10 = calculate_binomial_probabilities_n10(p)
        
        # Step 3: Create visualization for n=10 (if not skipped)
        if not args.skip_plots:
            create_probability_visualization_n10(probabilities_n10)
        
        # Step 4: Analyze normal approximation for n=100
        mu, sigma, prob_normal, prob_exact = analyze_normal_approximation_n100(p)
        
        # Step 5: Create comparison visualizations (if not skipped)
        if not args.skip_plots:
            create_comparison_visualization(p)
        
        # Step 6: Summary
        print("\n" + "=" * 60)
        print("📋 ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"✅ Successfully analyzed travel abroad data")
        print(f"📊 Key findings:")
        print(f"   • {percent_traveled:.2f}% of surveyed people traveled abroad")
        print(f"   • For n=10: Most likely scenario has probability {max(probabilities_n10.values()):.4f}")
        print(f"   • For n=100: Mean = {mu:.1f}, Std Dev = {sigma:.1f}")
        print(f"   • P(at least 59 out of 100) ≈ {prob_exact:.6f}")
        
        print(f"\n🎉 Analysis completed successfully!")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find data file at {args.data_path}")
        print(f"Please check the file path and try again.")
    except Exception as e:
        print(f"❌ An error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    main()