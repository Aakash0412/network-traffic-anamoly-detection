from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METRICS_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics"
)

FIGURES_DIR = (
    PROJECT_ROOT /
    "results" /
    "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load Supervised Results
# ============================================================

supervised_path = (
    METRICS_DIR /
    "supervised_ml_results.csv"
)

unsupervised_path = (
    METRICS_DIR /
    "unsupervised_ml_results.csv"
)


# ============================================================
# Supervised Model Comparison
# ============================================================

if supervised_path.exists():

    supervised = pd.read_csv(
        supervised_path
    )

    print("\n" + "=" * 70)
    print("SUPERVISED MODEL COMPARISON")
    print("=" * 70)

    print(
        supervised[
            [
                "model",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_roc_auc",
                "test_pr_auc"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # F1 Comparison
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.bar(
        supervised["model"],
        supervised["test_f1"]
    )

    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.title("Supervised Models - F1 Score")

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "supervised_f1_comparison.png",
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # ROC-AUC Comparison
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.bar(
        supervised["model"],
        supervised["test_roc_auc"]
    )

    plt.xlabel("Model")
    plt.ylabel("ROC-AUC")
    plt.title("Supervised Models - ROC-AUC")

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "supervised_roc_auc_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# Unsupervised Model Comparison
# ============================================================

if unsupervised_path.exists():

    unsupervised = pd.read_csv(
        unsupervised_path
    )

    print("\n" + "=" * 70)
    print("UNSUPERVISED MODEL COMPARISON")
    print("=" * 70)

    print(
        unsupervised.to_string(
            index=False
        )
    )

    plt.figure(figsize=(9, 6))

    plt.bar(
        unsupervised["model"],
        unsupervised["f1"]
    )

    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.title("Unsupervised Models - F1 Score")

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "unsupervised_f1_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# Combined Results
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Figures saved to: {FIGURES_DIR}"
)