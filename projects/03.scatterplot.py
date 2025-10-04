# Scatterplot Analysis - Convert from R to Python
# Original R notebook: 03.scatterplot.ipynb
# This creates: 1) Boxplot for iris sepal length, 2) Scatterplot for murders data

# Original R code was:
# library(dslabs)
# library(ggplot2)
# g <- ggplot(murders, aes(x = population / 10 ^6, y = total, label = abb)) + 
#     geom_point(aes(color = region)) + scale_x_log10() + scale_y_log10() + 
#     geom_text(size = 3, nudge_x = 0.075) + 
#     labs(title = "ScaterPlot", x = "population", y = "total")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sample murders data (since we don't have the dslabs package data)
def get_murders_data():
    # Create sample data similar to what was in the R script
    data = {
        'state': ['California', 'Texas', 'Florida', 'New York', 'Pennsylvania', 
                 'Illinois', 'Ohio', 'Georgia', 'North Carolina', 'Michigan'],
        'abb': ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI'],
        'region': ['West', 'South', 'South', 'Northeast', 'Northeast',
                  'North Central', 'North Central', 'South', 'South', 'North Central'],
        'population': [39500000, 29100000, 21500000, 20200000, 13000000,
                      12800000, 11800000, 10700000, 10400000, 10000000],
        'total': [1930, 1409, 1057, 774, 457, 664, 494, 376, 456, 413]
    }
    return pd.DataFrame(data)

# Part 1: Boxplot for iris sepal length
def create_iris_boxplot():
    print("Creating boxplot for iris sepal length...")
    
    # Load iris data
    iris = sns.load_dataset('iris')
    
    # Create boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=iris, y='sepal_length')
    
    # Add labels and title (like the R version)
    plt.title('Boxplot of Sepal Length - Iris Dataset')
    plt.ylabel('Sepal Length (cm)')
    plt.xlabel('Iris Dataset')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Part 2: Scatterplot for murders data
def create_murders_scatterplot():
    print("Creating scatterplot for murders data...")
    
    # Get the data
    murders = get_murders_data()
    
    # Create scatterplot with colors by region (like ggplot in R)
    plt.figure(figsize=(10, 6))
    
    # Plot points colored by region
    for region in murders['region'].unique():
        region_data = murders[murders['region'] == region]
        plt.scatter(region_data['population'] / 1e6,  # Convert to millions
                   region_data['total'], 
                   label=region, alpha=0.7)
        
        # Add state labels (like geom_text in R)
        for _, row in region_data.iterrows():
            plt.annotate(row['abb'], 
                        (row['population'] / 1e6, row['total']),
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=9)
    
    # Use log scale (like the R version)
    plt.xscale('log')
    plt.yscale('log')
    
    # Add labels and title (matching R labs() function)
    plt.title('ScaterPlot')  # Keeping the typo from original R code
    plt.xlabel('population')
    plt.ylabel('total')
    plt.legend(title='region')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Running scatterplot analysis...")
    
    # Run both parts like the original R notebook
    create_iris_boxplot()
    create_murders_scatterplot()
    
    print("Done!")