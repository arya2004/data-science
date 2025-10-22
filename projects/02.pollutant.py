import pandas as pd
import argparse

def analyze_data(file_path):
    """Performs pollutant dataset analysis (Python equivalent of the R script)."""

    # Load dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return

    print(f"Analyzing data from {file_path}...\n")

    # 1. Mean temperature for June (Month == 6)
    m = df[df['Month'] == 6]['Temp'].mean()
    print(f"1. Mean of temp when month is 6: {m:.1f}")

    # 2. Number of rows
    n = len(df)
    print(f"\n2. Number of rows: {n}")

    # 3. Last two rows
    print("\n3. Last two rows of the data:")
    print(df.tail(2).to_string())

    # 4. Ozone value in 47th row (1-indexed in R → iloc[46])
    p = df['Ozone'].iloc[46]
    print(f"\n4. Value of Ozone in 47th row: {p}")

    # 5. Missing values in Ozone column
    q = df['Ozone'].isna().sum()
    print(f"\n5. Number of missing values in Ozone column: {q}")

    # 6. Mean Ozone (excluding missing values)
    r = df['Ozone'].mean()
    print(f"\n6. Mean of Ozone column (excluding NAs): {r}")

    # 7. Mean Solar.R where Ozone > 31 and Temp > 90
    s = df[(df['Ozone'] > 31) & (df['Temp'] > 90)]['Solar.R'].mean()
    print(f"\n7. Mean of Solar.R where Ozone > 31 and Temp > 90: {s}")

    # 8. Maximum Ozone value in May
    t = df[df['Month'] == 5]['Ozone'].max()
    print(f"\n8. Maximum Ozone value in May: {t}")


def main():
    parser = argparse.ArgumentParser(description="Pollutant data analysis script.")
    parser.add_argument(
        '--file_path',
        type=str,
        default='../datasets/pollutant_csv.csv',
        help='Path to pollutant_csv.csv file.'
    )
    args = parser.parse_args()
    analyze_data(args.file_path)


if __name__ == "__main__":
    main()
