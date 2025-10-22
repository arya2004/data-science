"""
Lab1.py — Data Analysis and Visualization Script
------------------------------------------------

This script reproduces the analytical workflow of the R-based Lab1 notebook
using Python (pandas, seaborn, matplotlib).

It performs:
1. Exploratory data analysis on three CSV datasets:
   - Hair/Eye color data
   - Seed germination data
   - Air pollutant data

2. Visualization:
   - Boxplot using the Iris dataset
   - Scatterplot (using Penguins dataset as substitute for missing “murders” dataset)

Usage:
    python Lab1.py
    # or specify file paths explicitly:
    python Lab1.py --hair_eye_csv path/to/file.csv --germination_csv path/to/file.csv --pollutant_csv path/to/file.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys

# --- Dependency Check ---------------------------------------------------------
try:
    from pydataset import data  # optional dataset source
except ImportError:
    print("Error: 'pydataset' library not found.", file=sys.stderr)
    print("Please install it first by running: pip install pydataset", file=sys.stderr)
    sys.exit(1)


# -----------------------------------------------------------------------------
# SECTION 1: Hair/Eye Color Data Analysis
# -----------------------------------------------------------------------------
def analyze_hair_eye(file_path: str) -> None:
    """
    Analyze the hair_eye_color_csv.csv dataset.

    Tasks performed:
    1. Count number of people with brown eyes.
    2. Count number of people with blonde hair.
    3. Count number of brown-haired people with black eyes.
    4. Compute percentage of people with green eyes.
    5. Compute percentage of people with red hair and blue eyes.

    Parameters
    ----------
    file_path : str
        Path to the hair_eye_color_csv.csv file.
    """
    print(f"\n--- Analyzing Hair/Eye Color Data ({file_path}) ---")
    try:
        h_df = pd.read_csv(file_path, skipinitialspace=True)
        h_df.columns = h_df.columns.str.strip()  # Clean whitespace in headers
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        print("Current columns:", h_df.columns)
        return

    try:
        brown_eye_count = (h_df['Eye Color'] == "Brown").sum()
        print(f"Number of people have brown eye color: {brown_eye_count}")

        blonde_hair_count = (h_df['Hair Color'] == "Blonde").sum()
        print(f"Number of people have Blonde hair: {blonde_hair_count}")

        brown_hair_black_eye_count = (
            (h_df['Hair Color'] == "Brown") & (h_df['Eye Color'] == "Black")
        ).sum()
        print(f"Number of Brown haired people have Black eyes: {brown_hair_black_eye_count}")

        green_eye_percent = ((h_df['Eye Color'] == "Green").sum() * 100) / len(h_df)
        print(f"Percentage of people with Green eyes: {green_eye_percent:.0f}")

        red_blue_percent = (
            ((h_df['Hair Color'] == "Red") & (h_df['Eye Color'] == "Blue")).sum() * 100
        ) / len(h_df)
        print(f"Percentage of people have red hair and Blue eyes: {red_blue_percent:.0f}")

    except KeyError as e:
        print("\n--- ERROR ---")
        print(f"A column name in the script doesn't match the CSV file: {e}")
        print("Ensure columns are exactly:", h_df.columns.to_list())
        print("--- END ERROR ---")


# -----------------------------------------------------------------------------
# SECTION 2: Germination Data Analysis
# -----------------------------------------------------------------------------
def analyze_germination(file_path: str) -> None:
    """
    Analyze the germination_csv.csv dataset.

    Tasks performed:
    1. Compute average number of seeds germinated for uncovered boxes with watering level 4.
    2. Compute median number of seeds germinated for covered boxes.

    Parameters
    ----------
    file_path : str
        Path to the germination_csv.csv file.
    """
    print(f"\n--- Analyzing Germination Data ({file_path}) ---")
    try:
        seeds_df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return

    # Mean for uncovered boxes with watering level 4
    avg_uncovered_4 = seeds_df[
        (seeds_df['Box'] == "Uncovered") & (seeds_df['water_amt'] == 4)
    ]['germinated'].mean()
    print(f"1. Average number of seeds germinated for uncovered boxes with watering level 4: {avg_uncovered_4:.0f}")

    # Median for covered boxes
    median_covered = seeds_df[seeds_df['Box'] == "Covered"]['germinated'].median()
    print(f"2. Median value of germinated seeds for covered boxes: {median_covered:.0f}")


# -----------------------------------------------------------------------------
# SECTION 3: Pollutant Data Analysis
# -----------------------------------------------------------------------------
def analyze_pollutant(file_path: str) -> None:
    """
    Analyze the pollutant_csv.csv dataset.

    Tasks performed:
    1. Compute mean temperature for month == 6.
    2. Count total number of rows.
    3. Display last two rows.
    4. Retrieve value of Ozone in 47th row.
    5. Count missing values in Ozone column.
    6. Compute mean of Ozone column excluding NaN.
    7. Compute mean Solar.R for subset (Ozone > 31 and Temp > 90).
    8. Find maximum Ozone value in May (Month == 5).

    Parameters
    ----------
    file_path : str
        Path to the pollutant_csv.csv file.
    """
    print(f"\n--- Analyzing Pollutant Data ({file_path}) ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return

    m = df[df['Month'] == 6]['Temp'].mean()
    print(f"1. Mean of temp when month is 6: {m:.1f}")

    n = len(df)
    print(f"2. Number of rows: {n}")

    print("3. Last two rows of the data:")
    print(df.tail(2).to_string())

    ozone_47th = df['Ozone'].iloc[46]
    print(f"4. Value of Ozone in 47th row: {ozone_47th}")

    missing_ozone = df['Ozone'].isna().sum()
    print(f"5. Number of missing values in Ozone column: {missing_ozone}")

    mean_ozone = df['Ozone'].mean()
    print(f"6. Mean of Ozone column (excluding missing): {mean_ozone}")

    subset_mean_solar = df[(df['Ozone'] > 31) & (df['Temp'] > 90)]['Solar.R'].mean()
    print(f"7. Mean of Solar.R in subset (Ozone > 31, Temp > 90): {subset_mean_solar}")

    max_ozone_may = df[df['Month'] == 5]['Ozone'].max()
    print(f"8. Maximum ozone value in May: {max_ozone_may}")


# -----------------------------------------------------------------------------
# SECTION 4: Data Visualization
# -----------------------------------------------------------------------------
def plot_iris_boxplot() -> None:
    """
    Generate and save a boxplot of the Iris dataset.

    Visualization: Sepal Length distribution across Species.
    Output file: lab1_boxplot.png
    """
    print("\n--- Generating Iris Boxplot (saving as lab1_boxplot.png) ---")
    iris_df = sns.load_dataset("iris")

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=iris_df, x="sepal_length", y="species")

    plt.title("BoxPlot")
    plt.xlabel("Sepal Length")
    plt.ylabel("Species")

    plt.savefig("lab1_boxplot.png")
    plt.close()
    print("Saved plot to 'lab1_boxplot.png'")


def plot_murders_scatterplot() -> None:
    """
    Generate and save a scatterplot using the Penguins dataset
    (as a substitute for the unavailable 'murders' dataset).

    Visualization: Bill length vs. bill depth, colored by species.
    Output file: lab1_scatterplot.png
    """
    print("\n--- Generating Scatterplot (using penguins dataset) ---")
    penguins = sns.load_dataset("penguins").dropna()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=penguins,
        x='bill_length_mm',
        y='bill_depth_mm',
        hue='species',
        s=80
    )

    plt.title("Penguins Bill Dimensions (ScatterPlot)")
    plt.xlabel("Bill Length (mm)")
    plt.ylabel("Bill Depth (mm)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("lab1_scatterplot.png")
    plt.close()
    print("Saved plot to 'lab1_scatterplot.png'")


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Main entry point for script execution.

    Runs all data analyses and generates both visualizations.
    """
    parser = argparse.ArgumentParser(description="Python conversion of Lab1.ipynb")
    parser.add_argument('--hair_eye_csv', type=str, default='../datasets/hair_eye_color_csv.csv',
                        help='Path to hair_eye_color_csv.csv')
    parser.add_argument('--germination_csv', type=str, default='../datasets/germination_csv.csv',
                        help='Path to germination_csv.csv')
    parser.add_argument('--pollutant_csv', type=str, default='../datasets/pollutant_csv.csv',
                        help='Path to pollutant_csv.csv')

    args = parser.parse_args()

    analyze_hair_eye(args.hair_eye_csv)
    analyze_germination(args.germination_csv)
    analyze_pollutant(args.pollutant_csv)
    plot_iris_boxplot()
    plot_murders_scatterplot()

    print("\n--- Script execution complete ---")


if __name__ == "__main__":
    main()
