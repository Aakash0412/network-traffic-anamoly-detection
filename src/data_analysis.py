"""
ToN-IoT Network Dataset
Initial Dataset Analysis

Project:
Machine Learning Techniques for Real-Time Anomaly Detection
in Network Traffic
"""

from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train_test_network.csv"


# ============================================================
# Load Dataset
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Load the ToN-IoT network dataset."""

    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    print(f"Dataset path: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset loaded successfully.")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]:,}")

    return df


# ============================================================
# Dataset Overview
# ============================================================

def display_overview(df: pd.DataFrame) -> None:
    """Display basic information about the dataset."""

    print("\n" + "=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)

    print("\nShape:")
    print(df.shape)

    print("\nColumn Names:")
    for index, column in enumerate(df.columns, start=1):
        print(f"{index:2}. {column}")

    print("\nData Types:")
    print(df.dtypes)


# ============================================================
# Missing Value Analysis
# ============================================================

def analyze_missing_values(df: pd.DataFrame) -> None:
    """Analyze missing values."""

    print("\n" + "=" * 70)
    print("MISSING VALUE ANALYSIS")
    print("=" * 70)

    missing = df.isnull().sum()

    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("\nNo missing values found.")
        return

    print("\nColumns containing missing values:")
    print(missing)

    print("\nMissing-value percentage:")
    missing_percentage = (
        missing / len(df) * 100
    ).round(4)

    print(missing_percentage)


# ============================================================
# Duplicate Analysis
# ============================================================

def analyze_duplicates(df: pd.DataFrame) -> None:
    """Analyze duplicate records."""

    print("\n" + "=" * 70)
    print("DUPLICATE ANALYSIS")
    print("=" * 70)

    duplicate_count = df.duplicated().sum()

    print(f"\nDuplicate rows: {duplicate_count:,}")

    if duplicate_count > 0:
        percentage = duplicate_count / len(df) * 100
        print(f"Duplicate percentage: {percentage:.4f}%")
    else:
        print("No duplicate rows found.")


# ============================================================
# Target Analysis
# ============================================================

def analyze_target(df: pd.DataFrame) -> None:
    """Analyze the binary anomaly label."""

    print("\n" + "=" * 70)
    print("TARGET / LABEL ANALYSIS")
    print("=" * 70)

    if "label" not in df.columns:
        print("\nWARNING: 'label' column was not found.")
        return

    print("\nLabel values:")
    print(df["label"].value_counts(dropna=False))

    print("\nLabel percentages:")
    print(
        (df["label"].value_counts(normalize=True, dropna=False) * 100)
        .round(4)
    )

    print("\nLabel data type:")
    print(df["label"].dtype)


# ============================================================
# Attack Type Analysis
# ============================================================

def analyze_attack_types(df: pd.DataFrame) -> None:
    """Analyze the attack/traffic type column."""

    print("\n" + "=" * 70)
    print("ATTACK / TRAFFIC TYPE ANALYSIS")
    print("=" * 70)

    if "type" not in df.columns:
        print("\nWARNING: 'type' column was not found.")
        return

    print("\nNumber of unique types:")
    print(df["type"].nunique(dropna=False))

    print("\nType distribution:")
    print(df["type"].value_counts(dropna=False))

    print("\nType percentages:")
    print(
        (df["type"].value_counts(normalize=True, dropna=False) * 100)
        .round(4)
    )


# ============================================================
# Numerical Feature Analysis
# ============================================================

def analyze_numerical_features(df: pd.DataFrame) -> None:
    """Display numerical feature statistics."""

    print("\n" + "=" * 70)
    print("NUMERICAL FEATURE ANALYSIS")
    print("=" * 70)

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns

    print(f"\nNumber of numerical columns: {len(numerical_columns)}")

    print("\nNumerical columns:")
    print(list(numerical_columns))

    print("\nDescriptive statistics:")
    print(df[numerical_columns].describe().T)


# ============================================================
# Categorical Feature Analysis
# ============================================================

def analyze_categorical_features(df: pd.DataFrame) -> None:
    """Analyze categorical features."""

    print("\n" + "=" * 70)
    print("CATEGORICAL FEATURE ANALYSIS")
    print("=" * 70)

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    print(f"\nNumber of categorical columns: {len(categorical_columns)}")

    print("\nCategorical columns:")
    print(list(categorical_columns))

    print("\nUnique values per categorical column:")

    for column in categorical_columns:
        print(
            f"\n{column}: "
            f"{df[column].nunique(dropna=False)} unique values"
        )

        print(df[column].value_counts().head(10))


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Run the complete initial dataset analysis."""

    df = load_dataset()

    display_overview(df)

    analyze_missing_values(df)

    analyze_duplicates(df)

    analyze_target(df)

    analyze_attack_types(df)

    analyze_numerical_features(df)

    analyze_categorical_features(df)

    print("\n" + "=" * 70)
    print("INITIAL DATASET ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()