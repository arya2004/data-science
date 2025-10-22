"""
Pollutant Data Analysis (Python Version)
----------------------------------------

This script replicates the logic of the R script "02.ipynb" for pollutant data analysis.
It reads a CSV file and answers specific questions, including:
1. Mean temperature in June
2. Number of observations
3. Last two rows
4. Ozone value in 47th row
5. Count of missing Ozone values
6. Mean of Ozone excluding missing values
7. Mean Solar.R for subset where Ozone > 31 & Temp > 90
8. Maximum Ozone value in May

Author: Luckman Khan
Date: 2025-10-22
"""

import pandas as pd
import argparse
import sys


def analyze_pollutant_data(file_path: str) -> None:
    """
    Load pollutant CSV data and answer analysis questions.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing pollutant data.

    Notes
    -----
    - Column names must match exactly with the CSV: 'Ozone', 'Solar.R', 'Wind', 'Temp', 'Month', 'Day'.
    - Pandas is 0-based indexing; R is 1-based.
    - Mean calculations in Pandas automatically exclude NA values.
    """
    print(f"--- Analyzing Pollutant Data ({file_path}) ---")
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.\n")
        print("First 5 rows:")
        print(df.head().to_string())
        print("\nLast 5 rows:")
        print(df.tail().to_string())
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return

    # -----------------------------------------------------------------------------
    # Question 1: Mean Temp for June (Month == 6)
    # -----------------------------------------------------------------------------
    try:
        mean_temp_june = df.loc[df['Month'] == 6, 'Temp'].mean()
        print(f"\n1. Mean Temp for June: {mean_temp_june:.1f}")
    except KeyError as e:
        print(f"Column not found: {e}", file=sys.stderr)

    # -----------------------------------------------------------------------------
    # Question 2: Total number of observations
    # -----------------------------------------------------------------------------
    num_observations = len(df)
    print(f"\n2. Number of observations (rows): {num_observations}")

    # -----------------------------------------------------------------------------
    # Question 3: Last two rows
    # -----------------------------------------------------------------------------
    print("\n3. Last two rows of the data:")
    try:
        print(df.tail(2).to_string())
    except Exception as e:
        print(f"Error displaying last two rows: {e}", file=sys.stderr)

    # -----------------------------------------------------------------------------
    # Question 4: Ozone value in 47th row (R 1-based -> Python 0-based index 46)
    # -----------------------------------------------------------------------------
    try:
        ozone_47th = df.loc[46, 'Ozone']
        print(f"\n4. Ozone value in 47th row: {ozone_47th}")
    except IndexError:
        print(f"Row 47 is out of bounds (only {len(df)} rows).", file=sys.stderr)
    except KeyError as e:
        print(f"Column not found: {e}", file=sys.stderr)

    # -----------------------------------------------------------------------------
    # Question 5: Count of missing values in Ozone column
    # -----------------------------------------------------------------------------
    try:
        missing_ozone = df['Ozone'].isna().sum()
        print(f"\n5. Number of missing Ozone values: {missing_ozone}")
    except KeyError as e:
        print(f"Column not found: {e}", file=sys.stderr)

    # -----------------------------------------------------------------------------
    # Question 6: Mean of Ozone excluding missing values
    # -----------------------------------------------------------------------------
    try:
        mean_ozone = df['Ozone'].mean()
        print(f"\n6. Mean of Ozone (excluding NA): {mean_ozone}")
    except KeyError as e:
        print(f"Column not found: {e}", file=sys.stderr)

    # -----------------------------------------------------------------------------
    # Question 7: Subset where Ozone > 31 & Temp > 90, mean of Solar.R
    # -----------------------------------------------------------------------------
    try:
        subset_df = df[(df['Ozone'] > 31) & (df['Temp'] > 90)]
        print("\n7. Subset where Ozone > 31 and Temp > 90:")
        if not subset_df.empty:
            print(subset_df.to_string())
            mean_solar_r_subset = subset_df['Solar.R'].mean()
            print(f"   Mean Solar.R in subset: {mean_solar_r_subset:.1f}")
        else:
            print("   (Subset is empty; cannot calculate mean)")
    except KeyError as e:
        print(f"Column not found: {e}", file=sys.stderr)

    # -----------------------------------------------------------------------------
    # Question 8: Maximum Ozone value in May (Month == 5)
    # -----------------------------------------------------------------------------
    try:
        may_df = df[df['Month'] == 5]
        print("\n8. Data for May:")
        if not may_df.empty:
            # Display only first/last few rows for brevity if dataset is large
            if len(may_df) > 10:
                print(may_df.head().to_string())
                print("...")
                print(may_df.tail().to_string())
            else:
                print(may_df.to_string())

            max_ozone_may = may_df['Ozone'].max()
            print(f"   Maximum Ozone value in May: {max_ozone_may}")
        else:
            print("   (No data found for Month 5)")
    except KeyError as e:
        print(f"Column not found: {e}", file=sys.stderr)

    print("\n--- Analysis complete ---")


def main():
    """Parse command-line arguments and run pollutant analysis."""
    parser = argparse.ArgumentParser(
        description="Pollutant data analysis script (Python version of 02.ipynb)."
    )
    parser.add_argument(
        '--file_path',
        type=str,
        default='../datasets/pollutant_csv.csv',
        help='Path to pollutant CSV file.'
    )

    args = parser.parse_args()
    analyze_pollutant_data(args.file_path)


if __name__ == "__main__":
    main()
