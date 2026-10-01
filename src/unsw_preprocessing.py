from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "archive"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("UNSW-NB15 PREPROCESSING")
print("=" * 80)

print("\nLoading training data...")
train_df = pd.read_csv(TRAIN_FILE, low_memory=False)

print("Loading testing data...")
test_df = pd.read_csv(TEST_FILE, low_memory=False)

print(f"\nOriginal training shape: {train_df.shape}")
print(f"Original testing shape : {test_df.shape}")


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()


# ============================================================
# REMOVE DUPLICATES
# ============================================================

train_before = len(train_df)
test_before = len(test_df)

train_df = train_df.drop_duplicates().reset_index(drop=True)
test_df = test_df.drop_duplicates().reset_index(drop=True)

print(f"\nTraining duplicates removed: {train_before - len(train_df)}")
print(f"Testing duplicates removed : {test_before - len(test_df)}")


# ============================================================
# TARGET
# ============================================================

TARGET = "label"

y_train_full = train_df[TARGET].astype(int)
y_test = test_df[TARGET].astype(int)


# ============================================================
# DROP TARGET / NON-FEATURE COLUMNS
# ============================================================

DROP_COLUMNS = [
    "label",
    "attack_cat",
    "id",
]

X_train_full = train_df.drop(columns=DROP_COLUMNS)
X_test = test_df.drop(columns=DROP_COLUMNS)


# ============================================================
# REPLACE INF VALUES
# ============================================================

X_train_full = X_train_full.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = X_train_full.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train_full.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

print(f"\nNumerical features   : {len(numeric_features)}")
print(f"Categorical features : {len(categorical_features)}")


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

for col in numeric_features:
    median_value = X_train_full[col].median()

    X_train_full[col] = X_train_full[col].fillna(median_value)
    X_test[col] = X_test[col].fillna(median_value)


for col in categorical_features:
    X_train_full[col] = X_train_full[col].fillna("-")
    X_test[col] = X_test[col].fillna("-")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.1765,
    random_state=42,
    stratify=y_train_full,
)


print("\nSplit sizes:")
print(f"Train      : {len(X_train):,}")
print(f"Validation : {len(X_val):,}")
print(f"Test       : {len(X_test):,}")


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features,
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_features,
        ),
    ]
)


print("\nFitting preprocessor...")
X_train_processed = preprocessor.fit_transform(X_train)

print("Transforming validation set...")
X_val_processed = preprocessor.transform(X_val)

print("Transforming test set...")
X_test_processed = preprocessor.transform(X_test)


# ============================================================
# SAVE
# ============================================================

joblib.dump(
    preprocessor,
    MODEL_DIR / "unsw_preprocessor.joblib"
)

joblib.dump(
    X_train_processed,
    MODEL_DIR / "unsw_X_train.joblib"
)

joblib.dump(
    X_val_processed,
    MODEL_DIR / "unsw_X_val.joblib"
)

joblib.dump(
    X_test_processed,
    MODEL_DIR / "unsw_X_test.joblib"
)

joblib.dump(
    np.asarray(y_train),
    MODEL_DIR / "unsw_y_train.joblib"
)

joblib.dump(
    np.asarray(y_val),
    MODEL_DIR / "unsw_y_val.joblib"
)

joblib.dump(
    np.asarray(y_test),
    MODEL_DIR / "unsw_y_test.joblib"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("UNSW-NB15 PREPROCESSING COMPLETE")
print("=" * 80)

print(f"\nProcessed feature count: {X_train_processed.shape[1]}")

print(f"X_train: {X_train_processed.shape}")
print(f"X_val  : {X_val_processed.shape}")
print(f"X_test : {X_test_processed.shape}")

print("\nTraining label distribution:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nValidation label distribution:")
print(pd.Series(y_val).value_counts().sort_index())

print("\nTest label distribution:")
print(pd.Series(y_test).value_counts().sort_index())
