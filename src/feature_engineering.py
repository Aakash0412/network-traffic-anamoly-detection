from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train_test_network.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load Dataset
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Original shape: {df.shape}")


# ============================================================
# Remove Exact Duplicates
# ============================================================

df = df.drop_duplicates().reset_index(drop=True)

print(f"After duplicate removal: {df.shape}")


# ============================================================
# Target Separation
# ============================================================

y = df["label"]

X = df.drop(columns=["label", "type"])


# ============================================================
# Identify Numerical Features
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)


# ============================================================
# Log Transformation Analysis
# ============================================================
#
# Network traffic variables such as bytes, packets and
# duration can be highly right-skewed.
#
# We create transformed versions for analysis without
# replacing the original dataset yet.
# ============================================================

SKEWED_FEATURES = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "missed_bytes",
    "src_pkts",
    "src_ip_bytes",
    "dst_pkts",
    "dst_ip_bytes",
    "http_request_body_len",
    "http_response_body_len",
]


available_features = [
    feature
    for feature in SKEWED_FEATURES
    if feature in X.columns
]


for feature in available_features:

    X[f"{feature}_log"] = np.log1p(
        X[feature].clip(lower=0)
    )


print("\nLog-transformed features created:")

for feature in available_features:
    print(f"  {feature}_log")


# ============================================================
# Save Feature-Engineered Dataset
# ============================================================

output_path = (
    PROJECT_ROOT /
    "data" /
    "feature_engineered_network.csv"
)

feature_engineered = pd.concat(
    [X, y],
    axis=1
)

feature_engineered.to_csv(
    output_path,
    index=False
)

print("\nFeature engineering completed.")
print(f"Final shape: {feature_engineered.shape}")
print(f"Saved to: {output_path}")