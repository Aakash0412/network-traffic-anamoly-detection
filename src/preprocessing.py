from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train_test_network.csv"
MODEL_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42


# ============================================================
# Load Data
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Original dataset shape: {df.shape}")


# ============================================================
# Remove Duplicate Rows
# ============================================================

before = len(df)

df = df.drop_duplicates().reset_index(drop=True)

after = len(df)

print(f"Removed duplicates: {before - after:,}")
print(f"Dataset shape after duplicate removal: {df.shape}")


# ============================================================
# Remove Unwanted / High-Cardinality Columns
# ============================================================

DROP_COLUMNS = [
    "src_ip",
    "dst_ip",
    "dns_query",
    "ssl_cipher",
    "ssl_subject",
    "ssl_issuer",
    "http_uri",
    "http_user_agent",
    "http_orig_mime_types",
    "http_resp_mime_types",
    "type",
]

df = df.drop(columns=DROP_COLUMNS)

print("\nDropped columns:")
for column in DROP_COLUMNS:
    print(f"  - {column}")


# ============================================================
# Separate Features and Target
# ============================================================

X = df.drop(columns=["label"])
y = df["label"]

print(f"\nFeature shape: {X.shape}")
print(f"Target shape : {y.shape}")


# ============================================================
# Identify Feature Types
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# Train / Validation / Test Split
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE,
)

print("\nDataset split:")
print(f"Training   : {X_train.shape[0]:,}")
print(f"Validation : {X_val.shape[0]:,}")
print(f"Test       : {X_test.shape[0]:,}")


# ============================================================
# Preprocessing Pipeline
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_features,
        ),
    ]
)


# ============================================================
# Fit ONLY on Training Data
# ============================================================

print("\nFitting preprocessing pipeline...")

X_train_processed = preprocessor.fit_transform(X_train)

X_val_processed = preprocessor.transform(X_val)

X_test_processed = preprocessor.transform(X_test)


print("\nProcessed shapes:")
print(f"Training   : {X_train_processed.shape}")
print(f"Validation : {X_val_processed.shape}")
print(f"Test       : {X_test_processed.shape}")


# ============================================================
# Save Processed Data
# ============================================================

MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessor.joblib",
)

joblib.dump(
    X_train_processed,
    MODEL_DIR / "X_train.joblib",
)

joblib.dump(
    X_val_processed,
    MODEL_DIR / "X_val.joblib",
)

joblib.dump(
    X_test_processed,
    MODEL_DIR / "X_test.joblib",
)

joblib.dump(
    y_train,
    MODEL_DIR / "y_train.joblib",
)

joblib.dump(
    y_val,
    MODEL_DIR / "y_val.joblib",
)

joblib.dump(
    y_test,
    MODEL_DIR / "y_test.joblib",
)


print("\nPreprocessing completed successfully.")
print("Files saved inside the models/ directory.")