"""
Python Conversion of 01.ipynb (R Notebook)
==========================================
This script demonstrates basic DataFrame operations in Python using pandas,
equivalent to the R data frame operations in the original notebook.

Topics Covered:
- Creating DataFrames
- Accessing rows and columns
- DataFrame structure and summary statistics
- Adding columns and rows
- Filtering and conditional selection
- Using built-in datasets
- Basic plotting with matplotlib

Original notebook: 01.ipynb (R)
Converted to: 01.py (Python)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Set display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def main():
    print("=" * 80)
    print("Python Conversion of 01.ipynb - DataFrame Operations")
    print("=" * 80)
    print()
    
    # ========================================================================
    # Creating a DataFrame (equivalent to R data.frame)
    # ========================================================================
    print("1. Creating a DataFrame with cricket player statistics")
    print("-" * 80)
    
    matches_status = pd.DataFrame({
        'Name': ['Tendulkar', 'Ponting', 'Kallis', 'Dravid', 'Cook'],
        'Matches': [200, 168, 166, 164, 161],
        'Innings': [329, 287, 280, 286, 291],
        'HighestScore': [248, 257, 224, 270, 294],
        'Average': [53.78, 51.85, 55.37, 52.31, 45.35]
    })
    
    print(matches_status)
    print()
    
    # ========================================================================
    # DataFrame Structure and Information (equivalent to str() in R)
    # ========================================================================
    print("2. DataFrame Structure and Data Types")
    print("-" * 80)
    print(matches_status.info())
    print()
    print("Column names:", matches_status.columns.tolist())
    print()
    
    # Class/type of the object
    print(f"Type: {type(matches_status)}")
    print()
    
    # ========================================================================
    # Summary Statistics (equivalent to summary() in R)
    # ========================================================================
    print("3. Summary Statistics (5-point summary)")
    print("-" * 80)
    print(matches_status.describe())
    print()
    
    # ========================================================================
    # Accessing Individual Columns (equivalent to $ in R)
    # ========================================================================
    print("4. Accessing Individual Columns")
    print("-" * 80)
    print("Matches column:")
    print(matches_status['Matches'])
    print()
    
    # Individual columns are Series (equivalent to vectors in R)
    print(f"Type of column: {type(matches_status['Matches'])}")
    print()
    
    # ========================================================================
    # Column Operations
    # ========================================================================
    print("5. Column Operations")
    print("-" * 80)
    print(f"Sum of Average column: {matches_status['Average'].sum()}")
    print()
    
    # Accessing specific elements
    print(f"Average of player at index 1 (2nd player): {matches_status['Average'][1]}")
    print(f"Name at row 1, column 0: {matches_status.iloc[1, 0]}")
    print()
    
    # ========================================================================
    # DataFrame Dimensions
    # ========================================================================
    print("6. DataFrame Dimensions")
    print("-" * 80)
    print(f"Number of rows: {len(matches_status)}")
    print(f"Number of columns: {len(matches_status.columns)}")
    print(f"Shape: {matches_status.shape}")
    print()
    
    # ========================================================================
    # Head and Tail
    # ========================================================================
    print("7. Viewing First and Last Rows")
    print("-" * 80)
    print("First 5 rows (head):")
    print(matches_status.head())
    print()
    print("Last 5 rows (tail):")
    print(matches_status.tail())
    print()
    
    # ========================================================================
    # Slicing: Specific Rows and Columns
    # ========================================================================
    print("8. Slicing - Specific Rows and Columns")
    print("-" * 80)
    print("Rows 1-2 (index 1,2), Columns 2-3 (index 2,3):")
    print(matches_status.iloc[[1, 2], [2, 3]])
    print()
    
    # All rows, column 2
    print("All rows, column at index 2 (Innings):")
    print(matches_status.iloc[:, 2])
    print()
    
    # Row 4, all columns
    print("Row at index 4, all columns:")
    print(matches_status.iloc[4, :])
    print()
    
    # ========================================================================
    # Adding a New Column
    # ========================================================================
    print("9. Adding a New Column")
    print("-" * 80)
    matches_status['half_cent'] = [68, 62, 58, 63, 57]
    print("After adding 'half_cent' column:")
    print(matches_status)
    print()
    
    # ========================================================================
    # Adding New Rows (equivalent to rbind in R)
    # ========================================================================
    print("10. Adding New Rows")
    print("-" * 80)
    df1 = pd.DataFrame({
        'Name': ['Sangakara', 'Lara'],
        'Matches': [134, 144],
        'Innings': [233, 232],
        'HighestScore': [319, 400],
        'Average': [57.40, 52.80],
        'half_cent': [52, 48]
    })
    
    print("New data to add:")
    print(df1)
    print()
    
    # Concatenate (equivalent to rbind)
    matches_status = pd.concat([matches_status, df1], ignore_index=True)
    print("After adding new rows:")
    print(matches_status)
    print()
    
    # ========================================================================
    # Data Analysis Questions
    # ========================================================================
    print("11. Solving Analysis Questions")
    print("-" * 80)
    
    # Question 1: What is the highest score of Tendulkar?
    print("Q1: What is the highest score of Tendulkar?")
    tendulkar_highest = matches_status.loc[matches_status['Name'] == 'Tendulkar', 'HighestScore'].values[0]
    print(f"Answer: {tendulkar_highest}")
    print()
    
    # Question 2: Display the name and average of player with maximum highest score
    print("Q2: Display name and average of player with maximum highest score")
    max_score_idx = matches_status['HighestScore'].idxmax()
    max_highest_score = matches_status['HighestScore'].max()
    player_with_max = matches_status.loc[max_score_idx, ['Name', 'Average']]
    print(f"Maximum Highest Score: {max_highest_score}")
    print(player_with_max)
    print()
    
    # Question 3: Display name, matches, innings, and average of players with score > 250
    print("Q3: Players with highest score > 250")
    players_above_250 = matches_status[matches_status['HighestScore'] > 250][['Name', 'Matches', 'Innings', 'Average']]
    print(players_above_250)
    print()
    
    # Question 4: Find row numbers where highest score >= 270
    print("Q4: Row numbers where highest score >= 270")
    rows_above_270 = matches_status[matches_status['HighestScore'] >= 270].index.tolist()
    print(f"Row indices: {rows_above_270}")
    print()
    
    # Question 5: Modify Tendulkar's number of matches to 201
    print("Q5: Modify Tendulkar's matches to 201")
    tendulkar_idx = matches_status[matches_status['Name'] == 'Tendulkar'].index[0]
    print(f"Tendulkar's row index: {tendulkar_idx}")
    matches_status.loc[tendulkar_idx, 'Matches'] = 201
    print("After modification:")
    print(matches_status)
    print()
    
    # ========================================================================
    # Working with Built-in Datasets
    # ========================================================================
    print("12. Working with Built-in Datasets")
    print("-" * 80)
    
    # Iris dataset (equivalent to iris in R)
    print("Loading Iris dataset from seaborn:")
    iris = sns.load_dataset('iris')
    print(iris.head(10))
    print(f"\nIris dataset shape: {iris.shape}")
    print()
    
    # ========================================================================
    # Basic Visualization
    # ========================================================================
    print("13. Creating Visualizations")
    print("-" * 80)
    
    # Plot 1: Scatter plot of player statistics
    plt.figure(figsize=(10, 6))
    plt.scatter(matches_status['Matches'], matches_status['Average'], 
                c=matches_status['HighestScore'], cmap='viridis', s=200, alpha=0.7)
    plt.colorbar(label='Highest Score')
    plt.xlabel('Matches Played')
    plt.ylabel('Batting Average')
    plt.title('Cricket Player Statistics: Matches vs Average')
    
    # Add player names as labels
    for i, name in enumerate(matches_status['Name']):
        plt.annotate(name, (matches_status['Matches'].iloc[i], matches_status['Average'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('projects/01_player_stats.png', dpi=100, bbox_inches='tight')
    print("Saved: projects/01_player_stats.png")
    plt.close()
    
    # Plot 2: Iris dataset visualization (equivalent to ggplot in R)
    print("\nCreating Iris dataset visualization...")
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with species colors
    for species in iris['species'].unique():
        species_data = iris[iris['species'] == species]
        plt.scatter(species_data['sepal_length'], species_data['sepal_width'],
                   label=species, alpha=0.6, s=50)
    
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title('Iris Dataset: Sepal Length vs Sepal Width by Species')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('projects/01_iris_plot.png', dpi=100, bbox_inches='tight')
    print("Saved: projects/01_iris_plot.png")
    plt.close()
    
    print()
    print("=" * 80)
    print("Script execution completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
