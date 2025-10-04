#!/usr/bin/env python3
"""
Lab 3 Box Plot and Scatter Plot Analysis
=========================================

This script converts the R notebook (03.ipynb) to Python, implementing:
1. Box plot analysis using the iris dataset
2. Scatter plot analysis using the murders dataset

The script replicates the ggplot2 functionality from R using matplotlib and seaborn.

Author: Converted from R notebook
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path


def load_iris_data():
    """
    Load the iris dataset (equivalent to R's built-in iris dataset)
    
    Returns:
        pd.DataFrame: Iris dataset with columns for sepal/petal measurements and species
    """
    # Using seaborn's built-in iris dataset (same as R's iris)
    iris = sns.load_dataset('iris')
    
    # Rename columns to match R convention (capitalize first letters)
    iris.columns = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']
    
    # Capitalize species names to match R format
    iris['Species'] = iris['Species'].str.capitalize()
    
    return iris


def load_murders_data():
    """
    Load murders data (equivalent to R's dslabs murders dataset)
    
    Returns:
        pd.DataFrame: Murders dataset with state, population, and murder statistics
    """
    # Create sample data similar to the dslabs murders dataset
    # This matches the structure used in the R notebook
    data = {
        'state': ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
                 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
                 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
                 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
                 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
                 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
                 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
                 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia',
                 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'],
        'abb': ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
               'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
               'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
               'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
               'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'],
        'region': ['South', 'West', 'West', 'South', 'West', 'West',
                  'Northeast', 'South', 'South', 'South', 'West', 'West',
                  'North Central', 'North Central', 'North Central', 'North Central', 'South', 'South',
                  'Northeast', 'South', 'Northeast', 'North Central', 'North Central', 'South',
                  'North Central', 'West', 'North Central', 'West', 'Northeast', 'Northeast',
                  'West', 'Northeast', 'South', 'North Central', 'North Central',
                  'South', 'West', 'Northeast', 'Northeast', 'South',
                  'North Central', 'South', 'South', 'West', 'Northeast', 'South',
                  'West', 'South', 'North Central', 'West'],
        'population': [4779736, 710231, 6392017, 2915918, 37253956, 5029196,
                      3574097, 897934, 18801310, 9687653, 1360301, 1567582,
                      12830632, 6483802, 3046355, 2853118, 4339367, 4533372,
                      1328361, 5773552, 6547629, 9883640, 5303925, 2967297,
                      5988927, 989415, 1826341, 2700551, 1316470, 8791894,
                      2059179, 19378102, 9535483, 672591, 11536504,
                      3751351, 3831074, 12702379, 1052567, 4625364,
                      814180, 6346105, 25145561, 2763885, 625741, 8001024,
                      6724540, 1852994, 5686986, 563626],
        'total': [135, 19, 232, 93, 1257, 65, 42, 38, 669, 376,
                 7, 12, 364, 142, 21, 63, 116, 351,
                 11, 293, 118, 413, 53, 120,
                 321, 12, 32, 84, 5, 246,
                 56, 774, 286, 4, 310,
                 111, 36, 457, 16, 207,
                 8, 219, 805, 22, 2, 250,
                 156, 27, 97, 9]
    }
    
    return pd.DataFrame(data)


class BoxPlotAnalysis:
    """Class to handle all boxplot analysis from the R notebook"""
    
    def __init__(self, data):
        self.data = data
        self.figure_size = (10, 6)
    
    def basic_plot_mapping(self):
        """
        Equivalent to R: p <- ggplot(iris, aes(x = Petal.Length, y = Species))
        Shows the basic aesthetic mapping without geometry
        """
        plt.figure(figsize=self.figure_size)
        # This would just show the mapping, no actual plot in ggplot2
        plt.title("Basic Aesthetic Mapping (Petal Length vs Species)")
        plt.xlabel("Petal Length")
        plt.ylabel("Species")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def basic_boxplot(self):
        """
        Equivalent to R: ggplot(iris, aes(x = Sepal.Length, y = Species)) + geom_boxplot()
        """
        plt.figure(figsize=self.figure_size)
        
        # Create horizontal boxplot (y=Species means horizontal orientation)
        self.data.boxplot(column='Sepal.Length', by='Species', ax=plt.gca(), 
                         vert=False, patch_artist=False)
        
        plt.title("Basic Boxplot - Sepal Length by Species")
        plt.xlabel("Sepal Length")
        plt.ylabel("Species")
        plt.suptitle("")  # Remove the default title
        plt.tight_layout()
        plt.show()
    
    def colored_boxplot(self):
        """
        Equivalent to R: ggplot(..., fill=Species) + geom_boxplot()
        """
        plt.figure(figsize=self.figure_size)
        
        # Use seaborn for better color control
        sns.boxplot(data=self.data, y='Species', x='Sepal.Length', 
                   hue='Species', palette='Set2', legend=False)
        
        plt.title("Colored Boxplot - Sepal Length by Species")
        plt.xlabel("Sepal Length")
        plt.ylabel("Species")
        plt.tight_layout()
        plt.show()
    
    def boxplot_with_outliers(self):
        """
        Equivalent to R: geom_boxplot(outlier.shape = 4, outlier.color = "red", outlier.size = 4)
        """
        plt.figure(figsize=self.figure_size)
        
        # Create boxplot with custom outlier styling
        box_plot = sns.boxplot(data=self.data, y='Species', x='Sepal.Length', 
                              hue='Species', palette='Set2', legend=False)
        
        # Customize outliers (seaborn doesn't have direct outlier.shape=4 equivalent)
        # But we can modify the outlier properties
        for patch in box_plot.artists:
            patch.set_facecolor('lightblue')
        
        # Set outlier properties
        plt.setp(box_plot.findobj(plt.matplotlib.lines.Line2D), 
                color='red', marker='x', markersize=8)
        
        plt.title("Boxplot with Red Cross Outliers")
        plt.xlabel("Sepal Length")
        plt.ylabel("Species")
        plt.tight_layout()
        plt.show()
    
    def boxplot_no_legend(self):
        """
        Equivalent to R: theme(legend.position = "none")
        """
        plt.figure(figsize=self.figure_size)
        
        sns.boxplot(data=self.data, y='Species', x='Sepal.Length', 
                   hue='Species', palette='Set2', legend=False)
        
        plt.title("Boxplot without Legend")
        plt.xlabel("Sepal Length")
        plt.ylabel("Species")
        plt.tight_layout()
        plt.show()
    
    def boxplot_with_labels(self):
        """
        Equivalent to R: labs(title = "BoxPlot", x = "sepal length", y = "species")
        """
        plt.figure(figsize=self.figure_size)
        
        sns.boxplot(data=self.data, y='Species', x='Sepal.Length', 
                   hue='Species', palette='Set2', legend=False)
        
        # Custom labels matching the R version exactly
        plt.title("BoxPlot")
        plt.xlabel("sepal length")
        plt.ylabel("species")
        plt.tight_layout()
        plt.show()
    
    def rotated_boxplot(self):
        """
        Equivalent to R: coord_flip() - rotates the entire plot
        """
        plt.figure(figsize=self.figure_size)
        
        # coord_flip in ggplot2 swaps x and y axes
        sns.boxplot(data=self.data, x='Species', y='Sepal.Length', 
                   hue='Species', palette='Set2', legend=False)
        
        plt.title("BoxPlot")
        plt.xlabel("species")
        plt.ylabel("sepal length")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


class ScatterPlotAnalysis:
    """Class to handle all scatterplot analysis from the R notebook"""
    
    def __init__(self, data):
        self.data = data
        self.figure_size = (12, 8)
    
    def basic_mapping(self):
        """
        Equivalent to R: g <- ggplot(murders, aes(x = population / 10^6, y = total))
        """
        plt.figure(figsize=self.figure_size)
        
        # Just show the axes without points
        plt.xlabel("Population (millions)")
        plt.ylabel("Total Murders")
        plt.title("Basic Aesthetic Mapping")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def basic_scatterplot(self):
        """
        Equivalent to R: geom_point()
        """
        plt.figure(figsize=self.figure_size)
        
        x = self.data['population'] / 1e6  # Convert to millions
        y = self.data['total']
        
        plt.scatter(x, y, alpha=0.7)
        plt.xlabel("Population (millions)")
        plt.ylabel("Total Murders")
        plt.title("Basic Scatter Plot")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def log_scale_scatterplot(self):
        """
        Equivalent to R: scale_x_log10() + scale_y_log10()
        """
        plt.figure(figsize=self.figure_size)
        
        x = self.data['population'] / 1e6
        y = self.data['total']
        
        plt.scatter(x, y, alpha=0.7)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Population (millions)")
        plt.ylabel("Total Murders")
        plt.title("Log Scale Scatter Plot")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def colored_scatterplot(self):
        """
        Equivalent to R: aes(..., color = region)
        """
        plt.figure(figsize=self.figure_size)
        
        x = self.data['population'] / 1e6
        y = self.data['total']
        
        # Create scatter plot with colors by region
        for region in self.data['region'].unique():
            mask = self.data['region'] == region
            plt.scatter(x[mask], y[mask], label=region, alpha=0.7)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Population (millions)")
        plt.ylabel("Total Murders")
        plt.title("Colored by Region")
        plt.legend(title='Region')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def scatterplot_with_labels(self):
        """
        Equivalent to R: geom_text() with labels
        """
        plt.figure(figsize=self.figure_size)
        
        x = self.data['population'] / 1e6
        y = self.data['total']
        
        # Create scatter plot with colors by region
        for region in self.data['region'].unique():
            mask = self.data['region'] == region
            plt.scatter(x[mask], y[mask], label=region, alpha=0.7)
        
        # Add state abbreviation labels
        for i, row in self.data.iterrows():
            plt.annotate(row['abb'], 
                        (row['population'] / 1e6, row['total']),
                        xytext=(2, 2), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Population (millions)")
        plt.ylabel("Total Murders")
        plt.title("ScaterPlot")  # Keep the typo from original R code
        plt.legend(title='Region')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def scatterplot_nudged_text(self):
        """
        Equivalent to R: geom_text(size = 3, nudge_x = 0.075)
        """
        plt.figure(figsize=self.figure_size)
        
        x = self.data['population'] / 1e6
        y = self.data['total']
        
        # Create scatter plot with colors by region
        for region in self.data['region'].unique():
            mask = self.data['region'] == region
            plt.scatter(x[mask], y[mask], label=region, alpha=0.7)
        
        # Add state abbreviation labels with nudging
        for i, row in self.data.iterrows():
            # Nudge equivalent: offset the text position
            nudge_x = row['population'] / 1e6 * 0.075  # 7.5% nudge in x direction
            plt.annotate(row['abb'], 
                        (row['population'] / 1e6 + nudge_x, row['total']),
                        fontsize=8, alpha=0.8)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("population")
        plt.ylabel("total")
        plt.title("ScaterPlot")
        plt.legend(title='Region')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def run_full_analysis(show_plots=True):
    """
    Run the complete analysis matching the R notebook workflow
    
    Args:
        show_plots (bool): Whether to display plots (set to False for testing)
    """
    print("=" * 60)
    print("Lab 3: Box Plot and Scatter Plot Analysis")
    print("Converting R notebook (03.ipynb) to Python")
    print("=" * 60)
    
    # Load datasets
    print("\n1. Loading datasets...")
    iris_data = load_iris_data()
    murders_data = load_murders_data()
    
    print(f"   - Iris dataset: {iris_data.shape[0]} rows, {iris_data.shape[1]} columns")
    print(f"   - Murders dataset: {murders_data.shape[0]} rows, {murders_data.shape[1]} columns")
    
    if show_plots:
        # Part 1: Box Plot Analysis
        print("\n2. Running Box Plot Analysis...")
        print("   Following the step-by-step progression from the R notebook")
        
        box_analysis = BoxPlotAnalysis(iris_data)
        
        print("   - Basic aesthetic mapping")
        box_analysis.basic_plot_mapping()
        
        print("   - Basic boxplot")
        box_analysis.basic_boxplot()
        
        print("   - Colored boxplot")
        box_analysis.colored_boxplot()
        
        print("   - Boxplot with custom outliers")
        box_analysis.boxplot_with_outliers()
        
        print("   - Boxplot without legend")
        box_analysis.boxplot_no_legend()
        
        print("   - Boxplot with custom labels")
        box_analysis.boxplot_with_labels()
        
        print("   - Rotated boxplot (coord_flip)")
        box_analysis.rotated_boxplot()
        
        # Part 2: Scatter Plot Analysis
        print("\n3. Running Scatter Plot Analysis...")
        print("   Using murders dataset from dslabs equivalent")
        
        scatter_analysis = ScatterPlotAnalysis(murders_data)
        
        print("   - Basic aesthetic mapping")
        scatter_analysis.basic_mapping()
        
        print("   - Basic scatter plot")
        scatter_analysis.basic_scatterplot()
        
        print("   - Log scale scatter plot")
        scatter_analysis.log_scale_scatterplot()
        
        print("   - Colored by region")
        scatter_analysis.colored_scatterplot()
        
        print("   - With state labels")
        scatter_analysis.scatterplot_with_labels()
        
        print("   - With nudged text positioning")
        scatter_analysis.scatterplot_nudged_text()
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("All plots from the original R notebook have been recreated in Python")
    print("=" * 60)
    
    return iris_data, murders_data


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(
        description='Convert R notebook 03.ipynb to Python script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 03.py                    # Run full analysis with plots
  python 03.py --no-plots         # Run analysis without showing plots
  python 03.py --info             # Show dataset information only
        """
    )
    
    parser.add_argument('--no-plots', action='store_true',
                       help='Run analysis without displaying plots')
    parser.add_argument('--info', action='store_true',
                       help='Show dataset information only')
    
    args = parser.parse_args()
    
    if args.info:
        # Just show dataset info
        iris_data = load_iris_data()
        murders_data = load_murders_data()
        
        print("Dataset Information:")
        print("\nIris Dataset:")
        print(iris_data.head())
        print(f"Shape: {iris_data.shape}")
        
        print("\nMurders Dataset:")
        print(murders_data.head())
        print(f"Shape: {murders_data.shape}")
        
    else:
        # Run full analysis
        show_plots = not args.no_plots
        run_full_analysis(show_plots=show_plots)


if __name__ == "__main__":
    main()