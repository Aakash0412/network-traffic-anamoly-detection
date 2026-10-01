from pathlib import Path
import sys
import time

import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

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
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "UNSW-NB15": {
        "prefix": "unsw",
    },

    "CIC-IDS": {
        "prefix": "cic",
    },
}


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(dataset_name, prefix):

    print("\n" + "=" * 80)
    print(f"LOADING DATASET: {dataset_name}")
    print("=" * 80)

    X_train = joblib.load(
        MODEL_DIR / f"{prefix}_X_train.joblib"
    )

    X_val = joblib.load(
        MODEL_DIR / f"{prefix}_X_val.joblib"
    )

    X_test = joblib.load(
        MODEL_DIR / f"{prefix}_X_test.joblib"
    )

    y_train = joblib.load(
        MODEL_DIR / f"{prefix}_y_train.joblib"
    )

    y_val = joblib.load(
        MODEL_DIR / f"{prefix}_y_val.joblib"
    )

    y_test = joblib.load(
        MODEL_DIR / f"{prefix}_y_test.joblib"
    )

    print(f"X_train: {X_train.shape}")
    print(f"X_val  : {X_val.shape}")
    print(f"X_test : {X_test.shape}")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================
# GENERIC MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    model_name,
    dataset_name,
    X_test,
    y_test,
    training_time,
):

    print(f"\nEvaluating {model_name}...")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    inference_start = time.perf_counter()

    y_pred = model.predict(X_test)

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        y_score = model.predict_proba(
            X_test
        )[:, 1]

    elif hasattr(model, "decision_function"):

        y_score = model.decision_function(
            X_test
        )

    else:

        y_score = y_pred

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_score,
    )

    pr_auc = average_precision_score(
        y_test,
        y_score,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\nTest Results")
    print("-" * 50)

    print(f"Accuracy  : {accuracy:.6f}")
    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"F1 Score  : {f1:.6f}")
    print(f"ROC-AUC   : {roc_auc:.6f}")
    print(f"PR-AUC    : {pr_auc:.6f}")

    print("\nConfusion Matrix:")
    print(cm)

    print(
        f"\nTraining time : "
        f"{training_time:.4f} sec"
    )

    print(
        f"Inference time: "
        f"{inference_time:.4f} sec"
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "dataset": dataset_name,
        "model": model_name,

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,

        "training_time_sec": training_time,
        "inference_time_sec": inference_time,

        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    model,
    model_name,
    dataset_name,
    prefix,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):

    print("\n" + "-" * 80)
    print(f"{dataset_name} -> {model_name}")
    print("-" * 80)

    print(
        f"Training samples : {X_train.shape[0]:,}"
    )

    print(
        f"Training features: {X_train.shape[1]:,}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.perf_counter()
        - start
    )

    print(
        f"\nTraining completed in "
        f"{training_time:.4f} seconds"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nValidation")

    val_pred = model.predict(
        X_val
    )

    val_accuracy = accuracy_score(
        y_val,
        val_pred,
    )

    val_precision = precision_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    val_recall = recall_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    val_f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    print(
        f"Accuracy : {val_accuracy:.6f}"
    )

    print(
        f"Precision: {val_precision:.6f}"
    )

    print(
        f"Recall   : {val_recall:.6f}"
    )

    print(
        f"F1       : {val_f1:.6f}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    filename = (
        f"{prefix}_"
        f"{model_name.lower().replace(' ', '_')}.joblib"
    )

    model_path = (
        MODEL_DIR / filename
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nModel saved:"
        f"\n{model_path}"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    return evaluate_model(
        model=model,
        model_name=model_name,
        dataset_name=dataset_name,
        X_test=X_test,
        y_test=y_test,
        training_time=training_time,
    )


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def create_models():

    return {

        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),

        "SVM": LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            tol=1e-3,
            dual="auto",
            random_state=42,
        ),
    }


# ============================================================
# RUN NORMAL DATASET
# ============================================================

def run_dataset(
    dataset_name,
    prefix,
):

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_dataset(
        dataset_name,
        prefix,
    )

    models = create_models()

    results = []

    for model_name, model in models.items():

        result = train_model(
            model=model,
            model_name=model_name,
            dataset_name=dataset_name,
            prefix=prefix,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )

        results.append(result)

    return results


# ============================================================
# NORMALIZE OLD RESULT CSV
# ============================================================

def normalize_existing_results(
    existing_df
):

    # --------------------------------------------------------
    # Old ToN-IoT result file did not contain dataset
    # --------------------------------------------------------

    if "dataset" not in existing_df.columns:

        existing_df.insert(
            0,
            "dataset",
            "ToN-IoT",
        )

    # --------------------------------------------------------
    # Old -> new column names
    # --------------------------------------------------------

    rename_map = {

        "test_accuracy":
            "accuracy",

        "test_precision":
            "precision",

        "test_recall":
            "recall",

        "test_f1":
            "f1",

        "test_roc_auc":
            "roc_auc",

        "test_pr_auc":
            "pr_auc",

        "training_time_seconds":
            "training_time_sec",

        "inference_time_seconds":
            "inference_time_sec",
    }

    for old_column, new_column in rename_map.items():

        if (
            old_column in existing_df.columns
            and new_column not in existing_df.columns
        ):

            existing_df[new_column] = (
                existing_df[old_column]
            )

    # --------------------------------------------------------
    # Canonical schema
    # --------------------------------------------------------

    columns = [

        "dataset",
        "model",

        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",

        "training_time_sec",
        "inference_time_sec",

        "tn",
        "fp",
        "fn",
        "tp",
    ]

    # --------------------------------------------------------
    # Create missing columns
    # --------------------------------------------------------

    for column in columns:

        if column not in existing_df.columns:

            existing_df[column] = None

    return existing_df[
        columns
    ]


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results
):

    output_file = (
        RESULTS_DIR
        / "supervised_ml_results.csv"
    )

    new_results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Existing CSV
    # --------------------------------------------------------

    if output_file.exists():

        existing_df = pd.read_csv(
            output_file
        )

        existing_df = (
            normalize_existing_results(
                existing_df
            )
        )

        # ----------------------------------------------------
        # Remove previous result for same
        # dataset + model
        # ----------------------------------------------------

        for _, row in new_results_df.iterrows():

            existing_df = existing_df[
                ~(
                    (
                        existing_df["dataset"]
                        == row["dataset"]
                    )
                    &
                    (
                        existing_df["model"]
                        == row["model"]
                    )
                )
            ]

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        results_df = pd.concat(
            [
                existing_df,
                new_results_df,
            ],
            ignore_index=True,
        )

    else:

        results_df = new_results_df

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results_df = (
        results_df
        .sort_values(
            by=[
                "dataset",
                "model",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nResults saved to:"
        f"\n{output_file}"
    )

    return results_df


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results_df
):

    print("\n" + "=" * 110)
    print("SUPERVISED ML RESULTS")
    print("=" * 110)

    columns = [

        "dataset",
        "model",

        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",

        "training_time_sec",
    ]

    available_columns = [
        column
        for column in columns
        if column in results_df.columns
    ]

    print(
        results_df[
            available_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# CIC SVM ONLY
#
# Uses:
#   200,000 stratified training samples
#
# Evaluation:
#   Complete CIC validation set
#   Complete CIC test set
# ============================================================

def run_cic_svm_only():

    print("\n" + "#" * 80)
    print(
        "CIC-IDS SVM - 200K STRATIFIED TRAINING SAMPLE"
    )
    print("#" * 80)

    dataset_name = "CIC-IDS"

    prefix = (
        DATASETS[
            dataset_name
        ]["prefix"]
    )

    # --------------------------------------------------------
    # Load CIC data
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_dataset(
        dataset_name,
        prefix,
    )

    print("\nFULL CIC DATA")
    print("-" * 50)

    print(
        f"Training samples : "
        f"{X_train.shape[0]:,}"
    )

    print(
        f"Validation samples: "
        f"{X_val.shape[0]:,}"
    )

    print(
        f"Test samples      : "
        f"{X_test.shape[0]:,}"
    )

    print(
        f"Features           : "
        f"{X_train.shape[1]}"
    )

    # --------------------------------------------------------
    # Sample size
    # --------------------------------------------------------

    SAMPLE_SIZE = 200_000

    print("\n" + "=" * 80)
    print("CREATING STRATIFIED SVM SAMPLE")
    print("=" * 80)

    # --------------------------------------------------------
    # Stratified sample
    # --------------------------------------------------------

    if X_train.shape[0] > SAMPLE_SIZE:

        X_svm, _, y_svm, _ = (
            train_test_split(
                X_train,
                y_train,
                train_size=SAMPLE_SIZE,
                stratify=y_train,
                random_state=42,
            )
        )

    else:

        X_svm = X_train
        y_svm = y_train

    print(
        f"\nSVM training samples: "
        f"{X_svm.shape[0]:,}"
    )

    print(
        f"SVM features: "
        f"{X_svm.shape[1]}"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_counts = (
        pd.Series(y_svm)
        .value_counts()
        .sort_index()
    )

    print(
        "\nSVM training class distribution:"
    )

    print(
        class_counts.to_string()
    )

    class_percentages = (
        pd.Series(y_svm)
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    print(
        "\nSVM training class percentages:"
    )

    for label, percentage in (
        class_percentages.items()
    ):

        print(
            f"Class {label}: "
            f"{percentage:.2f}%"
        )

    # --------------------------------------------------------
    # Create Linear SVM
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("INITIALIZING LINEAR SVM")
    print("=" * 80)

    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
        tol=1e-3,
        dual="auto",
        random_state=42,
    )

    print(
        "Estimator : LinearSVC"
    )

    print(
        "C         : 1.0"
    )

    print(
        "Class weight: balanced"
    )

    print(
        "Max iterations: 3000"
    )

    print(
        "Tolerance : 0.001"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STARTING SVM TRAINING")
    print("=" * 80)

    print(
        "Training ONLY on 200,000 stratified samples."
    )

    start = time.perf_counter()

    model.fit(
        X_svm,
        y_svm,
    )

    training_time = (
        time.perf_counter()
        - start
    )

    print(
        f"\nSVM training completed in "
        f"{training_time:.4f} seconds"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    val_pred = model.predict(
        X_val
    )

    val_score = (
        model.decision_function(
            X_val
        )
    )

    val_accuracy = accuracy_score(
        y_val,
        val_pred,
    )

    val_precision = precision_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    val_recall = recall_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    val_f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    val_roc_auc = roc_auc_score(
        y_val,
        val_score,
    )

    val_pr_auc = (
        average_precision_score(
            y_val,
            val_score,
        )
    )

    print(
        f"Validation Accuracy : "
        f"{val_accuracy:.6f}"
    )

    print(
        f"Validation Precision: "
        f"{val_precision:.6f}"
    )

    print(
        f"Validation Recall   : "
        f"{val_recall:.6f}"
    )

    print(
        f"Validation F1       : "
        f"{val_f1:.6f}"
    )

    print(
        f"Validation ROC-AUC  : "
        f"{val_roc_auc:.6f}"
    )

    print(
        f"Validation PR-AUC   : "
        f"{val_pr_auc:.6f}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "cic_svm.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nModel saved to:"
        f"\n{model_path}"
    )

    # --------------------------------------------------------
    # Full test set
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL TEST - FULL CIC-IDS TEST SET")
    print("=" * 80)

    print(
        f"Testing on "
        f"{X_test.shape[0]:,} samples."
    )

    inference_start = (
        time.perf_counter()
    )

    y_pred = model.predict(
        X_test
    )

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    y_score = (
        model.decision_function(
            X_test
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_score,
    )

    pr_auc = (
        average_precision_score(
            y_test,
            y_score,
        )
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print("\nFINAL TEST RESULTS")
    print("-" * 60)

    print(
        f"Accuracy  : {accuracy:.6f}"
    )

    print(
        f"Precision : {precision:.6f}"
    )

    print(
        f"Recall    : {recall:.6f}"
    )

    print(
        f"F1 Score  : {f1:.6f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.6f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.6f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    print(
        f"\nTraining time : "
        f"{training_time:.4f} sec"
    )

    print(
        f"Inference time: "
        f"{inference_time:.4f} sec"
    )

    # --------------------------------------------------------
    # Result dictionary
    # --------------------------------------------------------

    result = {

        "dataset": dataset_name,

        "model": "SVM",

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": roc_auc,

        "pr_auc": pr_auc,

        "training_time_sec":
            training_time,

        "inference_time_sec":
            inference_time,

        "tn": int(cm[0, 0]),

        "fp": int(cm[0, 1]),

        "fn": int(cm[1, 0]),

        "tp": int(cm[1, 1]),
    }

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results_df = save_results(
        [result]
    )

    print("\n" + "=" * 80)
    print("CIC-IDS SVM COMPLETE")
    print("=" * 80)

    print_results(
        results_df[
            results_df["dataset"]
            == "CIC-IDS"
        ]
    )


# ============================================================
# FULL SUPERVISED ML
#
# NOTE:
# This function trains all three models on UNSW/CIC.
# Do NOT use this for the current SVM experiment.
# Use --cic-svm instead.
# ============================================================

def run_full_supervised_ml():

    print("\n" + "#" * 80)
    print("SUPERVISED MACHINE LEARNING")
    print("#" * 80)

    all_results = []

    for dataset_name, config in (
        DATASETS.items()
    ):

        results = run_dataset(
            dataset_name,
            config["prefix"],
        )

        all_results.extend(
            results
        )

    results_df = save_results(
        all_results
    )

    print_results(
        results_df
    )

    print("\n" + "=" * 80)
    print("SUPERVISED ML COMPLETE")
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # CIC SVM ONLY
    # --------------------------------------------------------

    if (
        len(sys.argv) > 1
        and sys.argv[1] == "--cic-svm"
    ):

        run_cic_svm_only()

    # --------------------------------------------------------
    # Otherwise run everything
    # --------------------------------------------------------

    else:

        run_full_supervised_ml()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()