from pathlib import Path
import time
import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load Preprocessed Data
# ============================================================

print("Loading preprocessed data...")

X_train = joblib.load(MODEL_DIR / "X_train.joblib")
X_val = joblib.load(MODEL_DIR / "X_val.joblib")
X_test = joblib.load(MODEL_DIR / "X_test.joblib")

y_train = joblib.load(MODEL_DIR / "y_train.joblib")
y_val = joblib.load(MODEL_DIR / "y_val.joblib")
y_test = joblib.load(MODEL_DIR / "y_test.joblib")

print(f"Training data   : {X_train.shape}")
print(f"Validation data : {X_val.shape}")
print(f"Test data       : {X_test.shape}")


# ============================================================
# Evaluation Function
# ============================================================

def evaluate_model(model, model_name):

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(f"Training completed in {training_time:.2f} seconds")

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_predictions = model.predict(X_val)

    if hasattr(model, "predict_proba"):
        val_scores = model.predict_proba(X_val)[:, 1]
    else:
        val_scores = model.decision_function(X_val)

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision = precision_score(
        y_val, val_predictions, zero_division=0
    )
    val_recall = recall_score(
        y_val, val_predictions, zero_division=0
    )
    val_f1 = f1_score(
        y_val, val_predictions, zero_division=0
    )
    val_roc_auc = roc_auc_score(y_val, val_scores)
    val_pr_auc = average_precision_score(y_val, val_scores)

    print("\nValidation Results:")
    print(f"Accuracy  : {val_accuracy:.4f}")
    print(f"Precision : {val_precision:.4f}")
    print(f"Recall    : {val_recall:.4f}")
    print(f"F1 Score  : {val_f1:.4f}")
    print(f"ROC-AUC   : {val_roc_auc:.4f}")
    print(f"PR-AUC    : {val_pr_auc:.4f}")

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_start = time.time()

    test_predictions = model.predict(X_test)

    inference_time = time.time() - test_start

    if hasattr(model, "predict_proba"):
        test_scores = model.predict_proba(X_test)[:, 1]
    else:
        test_scores = model.decision_function(X_test)

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(
        y_test, test_predictions, zero_division=0
    )
    test_recall = recall_score(
        y_test, test_predictions, zero_division=0
    )
    test_f1 = f1_score(
        y_test, test_predictions, zero_division=0
    )
    test_roc_auc = roc_auc_score(y_test, test_scores)
    test_pr_auc = average_precision_score(y_test, test_scores)

    cm = confusion_matrix(y_test, test_predictions)

    print("\nTest Results:")
    print(f"Accuracy  : {test_accuracy:.4f}")
    print(f"Precision : {test_precision:.4f}")
    print(f"Recall    : {test_recall:.4f}")
    print(f"F1 Score  : {test_f1:.4f}")
    print(f"ROC-AUC   : {test_roc_auc:.4f}")
    print(f"PR-AUC    : {test_pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nInference time: {inference_time:.4f} seconds")

    # --------------------------------------------------------
    # Save Model
    # --------------------------------------------------------

    filename = model_name.lower().replace(" ", "_") + ".joblib"

    joblib.dump(
        model,
        MODEL_DIR / filename
    )

    print(f"Model saved: models/{filename}")

    # --------------------------------------------------------
    # Save Metrics
    # --------------------------------------------------------

    metrics = {
        "model": model_name,

        "validation_accuracy": val_accuracy,
        "validation_precision": val_precision,
        "validation_recall": val_recall,
        "validation_f1": val_f1,
        "validation_roc_auc": val_roc_auc,
        "validation_pr_auc": val_pr_auc,

        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,

        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,

        "tn": cm[0][0],
        "fp": cm[0][1],
        "fn": cm[1][0],
        "tp": cm[1][1],
    }

    return metrics


# ============================================================
# Models
# ============================================================

models = [

    # --------------------------------------------------------
    # 1. Decision Tree
    # --------------------------------------------------------

    (
        "Decision Tree",
        DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        )
    ),

    # --------------------------------------------------------
    # 2. Random Forest
    # --------------------------------------------------------

    (
        "Random Forest",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
    ),

    # --------------------------------------------------------
    # 3. Support Vector Machine
    # --------------------------------------------------------

    (
        "SVM",
        SVC(
            kernel="linear",
            probability=False,
            class_weight="balanced",
            random_state=42
        )
    ),
]


# ============================================================
# Train Models
# ============================================================

all_results = []

for model_name, model in models:

    results = evaluate_model(
        model,
        model_name
    )

    all_results.append(results)


# ============================================================
# Save Summary
# ============================================================

results_df = pd.DataFrame(all_results)

results_path = RESULTS_DIR / "supervised_ml_results.csv"

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# Display Final Results
# ============================================================

print("\n" + "=" * 70)
print("ALL SUPERVISED MODELS COMPLETED")
print("=" * 70)

print(
    results_df[
        [
            "model",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
            "test_pr_auc",
        ]
    ].to_string(index=False)
)

print(f"\nResults saved to: {results_path}")