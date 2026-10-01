from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "archive-2"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND CIC FILES
# ============================================================

CIC_FILES = sorted(DATA_DIR.glob("*.csv"))

print("=" * 80)
print("CIC-IDS PREPROCESSING")
print("=" * 80)

print(f"\nFound {len(CIC_FILES)} CIC files:")

for file in CIC_FILES:
    print(f"  - {file.name}")


# ============================================================
# LOAD FILES
# ============================================================

dataframes = []

for file in CIC_FILES:

    print(f"\nLoading: {file.name}")

    df = pd.read_csv(
        file,
        low_memory=False
    )

    print(f"Shape: {df.shape}")

    # Normalize column names
    df.columns = df.columns.str.strip()

    dataframes.append(df)


# ============================================================
# COMBINE DATASETS
# ============================================================

print("\nCombining CIC files...")

df = pd.concat(
    dataframes,
    ignore_index=True
)

del dataframes

print(f"Combined shape: {df.shape}")


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates().reset_index(drop=True)

duplicates_removed = before_duplicates - len(df)

print(f"\nDuplicates removed: {duplicates_removed}")
print(f"Shape after duplicates: {df.shape}")


# ============================================================
# TARGET
# ============================================================

TARGET = "Label"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )


print("\nOriginal label distribution:")
print(df[TARGET].value_counts(dropna=False))


# ============================================================
# NORMALIZE LABEL
# ============================================================

df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.strip()
)

# BENIGN = 0
# Everything else = 1

y = (
    df[TARGET]
    .str.upper()
    .ne("BENIGN")
    .astype(int)
)


print("\nBinary label distribution:")

print(
    pd.Series(y)
    .value_counts()
    .sort_index()
    .rename(index={
        0: "Normal / BENIGN",
        1: "Attack"
    })
)


# ============================================================
# DROP TARGET
# ============================================================

X = df.drop(columns=[TARGET])


# ============================================================
# REMOVE INF VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# REMOVE COMPLETELY EMPTY / CONSTANT COLUMNS
# ============================================================

empty_columns = [
    col
    for col in X.columns
    if X[col].isna().all()
]

if empty_columns:
    print("\nRemoving completely empty columns:")
    print(empty_columns)

    X = X.drop(columns=empty_columns)


constant_columns = [
    col
    for col in X.columns
    if X[col].nunique(dropna=False) <= 1
]

if constant_columns:
    print("\nRemoving constant columns:")
    print(constant_columns)

    X = X.drop(columns=constant_columns)


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()


print("\nFeature types:")
print(f"Numerical features   : {len(numeric_features)}")
print(f"Categorical features : {len(categorical_features)}")


if categorical_features:
    print("\nCategorical columns:")
    for col in categorical_features:
        print(f"  - {col}")


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print("\nHandling missing values...")

for col in numeric_features:

    median_value = X[col].median()

    X[col] = X[col].fillna(
        median_value
    )


for col in categorical_features:

    X[col] = X[col].fillna("-")


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\nCreating train/validation/test split...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

del X_temp
del y_temp


print("\nSplit sizes:")
print(f"Train      : {len(X_train):,}")
print(f"Validation : {len(X_val):,}")
print(f"Test       : {len(X_test):,}")


# ============================================================
# PREPROCESSOR
# ============================================================

print("\nCreating preprocessing pipeline...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ============================================================
# FIT ONLY ON TRAINING DATA
# ============================================================

print("\nFitting preprocessor on training data...")

X_train_processed = preprocessor.fit_transform(
    X_train
)


print("Transforming validation data...")

X_val_processed = preprocessor.transform(
    X_val
)


print("Transforming test data...")

X_test_processed = preprocessor.transform(
    X_test
)


# ============================================================
# SAVE PREPROCESSOR
# ============================================================

joblib.dump(
    preprocessor,
    MODEL_DIR / "cic_preprocessor.joblib"
)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

joblib.dump(
    X_train_processed,
    MODEL_DIR / "cic_X_train.joblib"
)

joblib.dump(
    X_val_processed,
    MODEL_DIR / "cic_X_val.joblib"
)

joblib.dump(
    X_test_processed,
    MODEL_DIR / "cic_X_test.joblib"
)

joblib.dump(
    np.asarray(y_train),
    MODEL_DIR / "cic_y_train.joblib"
)

joblib.dump(
    np.asarray(y_val),
    MODEL_DIR / "cic_y_val.joblib"
)

joblib.dump(
    np.asarray(y_test),
    MODEL_DIR / "cic_y_test.joblib"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("CIC-IDS PREPROCESSING COMPLETE")
print("=" * 80)

print(f"\nProcessed feature count: {X_train_processed.shape[1]}")

print(
    f"X_train: {X_train_processed.shape}"
)

print(
    f"X_val  : {X_val_processed.shape}"
)

print(
    f"X_test : {X_test_processed.shape}"
)


print("\nTraining label distribution:")

print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)


print("\nValidation label distribution:")

print(
    pd.Series(y_val)
    .value_counts()
    .sort_index()
)


print("\nTest label distribution:")

print(
    pd.Series(y_test)
    .value_counts()
    .sort_index()
)