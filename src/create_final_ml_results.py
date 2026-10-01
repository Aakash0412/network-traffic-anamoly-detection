from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "metrics"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "final_classical_ml_results.csv"
)


# ============================================================
# LOAD RESULTS
# ============================================================

supervised_file = (
    RESULTS_DIR
    / "supervised_ml_results.csv"
)

unsupervised_file = (
    RESULTS_DIR
    / "unsupervised_ml_results.csv"
)


print("=" * 100)
print("CREATING FINAL CLASSICAL ML RESULTS")
print("=" * 100)


# ============================================================
# SUPERVISED
# ============================================================

print("\nLoading supervised results...")

supervised = pd.read_csv(
    supervised_file
)

print(
    f"Supervised experiments: "
    f"{len(supervised)}"
)


# ============================================================
# UNSUPERVISED
# ============================================================

print("\nLoading unsupervised results...")

unsupervised = pd.read_csv(
    unsupervised_file
)

print(
    f"Unsupervised experiments: "
    f"{len(unsupervised)}"
)


# ============================================================
# ADD CATEGORY
# ============================================================

supervised["category"] = (
    "Supervised"
)

unsupervised["category"] = (
    "Unsupervised / Semi-supervised"
)


# ============================================================
# COMMON COLUMNS
# ============================================================

common_columns = [

    "category",

    "dataset",

    "model",

    "accuracy",

    "precision",

    "recall",

    "f1",

    "roc_auc",

    "pr_auc",

    "ari",

    "training_time_sec",

    "inference_time_sec",

    "tn",

    "fp",

    "fn",

    "tp",
]


# Add missing columns where necessary

for column in common_columns:

    if column not in supervised.columns:

        supervised[column] = pd.NA

    if column not in unsupervised.columns:

        unsupervised[column] = pd.NA


# ============================================================
# COMBINE
# ============================================================

final_df = pd.concat(
    [
        supervised[common_columns],
        unsupervised[common_columns],
    ],
    ignore_index=True,
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

final_df = (
    final_df
    .drop_duplicates(
        subset=[
            "dataset",
            "model",
        ],
        keep="last",
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# SORT
# ============================================================

dataset_order = [
    "ToN-IoT",
    "UNSW-NB15",
    "CIC-IDS",
]

category_order = [
    "Supervised",
    "Unsupervised / Semi-supervised",
]

final_df["dataset"] = pd.Categorical(
    final_df["dataset"],
    categories=dataset_order,
    ordered=True,
)

final_df["category"] = pd.Categorical(
    final_df["category"],
    categories=category_order,
    ordered=True,
)

final_df = (
    final_df
    .sort_values(
        [
            "dataset",
            "category",
            "model",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# SAVE
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("FINAL CLASSICAL ML RESULTS")
print("=" * 100)

display_columns = [

    "category",
    "dataset",
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "ari",
]

print(
    final_df[
        display_columns
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 100)

print(
    f"Total experiments: "
    f"{len(final_df)}"
)

print(
    f"Results saved to:\n"
    f"{OUTPUT_FILE}"
)

print("=" * 100)