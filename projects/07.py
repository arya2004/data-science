import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import argparse
import sys


def normalize_column(column: pd.Series) -> pd.Series:
    """
    Min-max normalizes a pandas Series.
    
    If all values are the same, returns a Series of 0.0 to avoid division by zero.
    """
    min_val = column.min()
    max_val = column.max()
    if max_val == min_val:
        return pd.Series(0.0, index=column.index)
    return (column - min_val) / (max_val - min_val)


def main():
    """
    Loads WBC dataset, normalizes features, splits train/test sets, performs KNN classification,
    and prints evaluation metrics including confusion matrix, accuracy, sensitivity, specificity, and precision.
    """
    parser = argparse.ArgumentParser(description="KNN classification on WBC dataset.")
    parser.add_argument('--file_path', type=str, default='../datasets/wbc_csv.csv',
                        help='Path to the CSV file containing WBC data.')
    parser.add_argument('--k_neighbors', type=int, default=3,
                        help='Number of neighbors (k) for KNN.')
    parser.add_argument('--random_seed', type=int, default=123,
                        help='Random seed for shuffling data.')
    parser.add_argument('--train_split_index', type=int, default=469,
                        help='Index to split train and test sets.')

    args = parser.parse_args()

    # Load dataset
    try:
        df = pd.read_csv(args.file_path)
        print(f"Data loaded successfully from {args.file_path}.")
        print("First 5 rows:")
        print(df.head().to_string())
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert diagnosis to categorical type
    try:
        df['diagnosis'] = df['diagnosis'].astype('category')
        print("\nConverted 'diagnosis' column to categorical.")
    except KeyError:
        print("Error: 'diagnosis' column not found.", file=sys.stderr)
        sys.exit(1)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    print(f"\nData shuffled with seed {args.random_seed}.")

    # Select feature columns (R columns 3:32 → Python indices 2:31)
    try:
        feature_columns = df.columns[2:32]
        df_features = df[feature_columns]
        print(f"Selected {len(feature_columns)} feature columns for normalization.")
    except IndexError:
        print("Error: Not enough columns to select features.", file=sys.stderr)
        sys.exit(1)

    # Normalize features using min-max scaling
    df_normalized = df_features.apply(normalize_column)
    print("\nNormalization complete. First 5 rows:")
    print(df_normalized.head().to_string())

    # Split data into training and test sets
    split_idx = args.train_split_index
    num_rows = len(df_normalized)
    test_end_idx = min(split_idx + 100, num_rows)  # R used 470:569 (100 rows)
    if split_idx >= num_rows or split_idx < 0:
        print(f"Error: train_split_index {split_idx} out of bounds.", file=sys.stderr)
        sys.exit(1)

    train_features = df_normalized.iloc[:split_idx, :]
    test_features = df_normalized.iloc[split_idx:test_end_idx, :]
    train_labels = df.iloc[:split_idx, 1]
    test_labels = df.iloc[split_idx:test_end_idx, 1]

    print(f"\nTraining set: {len(train_features)} samples")
    print(f"Test set: {len(test_features)} samples")

    # Train KNN classifier
    try:
        knn = KNeighborsClassifier(n_neighbors=args.k_neighbors)
        knn.fit(train_features, train_labels)
        print(f"\nTrained k-NN model with k={args.k_neighbors}.")
    except Exception as e:
        print(f"Error training KNN model: {e}", file=sys.stderr)
        sys.exit(1)

    # Predict on test set
    predictions = knn.predict(test_features)
    print("\nPredictions complete.")

    # Evaluate performance
    try:
        cm = confusion_matrix(test_labels, predictions, labels=knn.classes_)
        cm_df = pd.DataFrame(cm, index=knn.classes_, columns=knn.classes_)
        print("\nConfusion Matrix (Rows: Actual, Columns: Predicted):")
        print(cm_df)

        # Extract TP, TN, FP, FN for binary classification
        TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

        # Compute metrics
        total = TN + TP + FN + FP
        accuracy = (TP + TN) / total if total else 0
        sensitivity = TP / (TP + FN) if (TP + FN) else 0
        specificity = TN / (TN + FP) if (TN + FP) else 0
        precision = TP / (TP + FP) if (TP + FP) else 0

        print(f"\nTN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")
        print(f"\nAccuracy: {accuracy:.6f}")
        print(f"Sensitivity (Recall): {sensitivity:.6f}")
        print(f"Specificity: {specificity:.6f}")
        print(f"Precision: {precision:.6f}")

    except Exception as e:
        print(f"Error computing metrics: {e}", file=sys.stderr)

    print("\n--- Script finished ---")


if __name__ == "__main__":
    main()
