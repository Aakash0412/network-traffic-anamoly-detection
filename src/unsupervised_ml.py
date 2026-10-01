from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    adjusted_rand_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results" / "metrics"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Computationally manageable for an 8 GB RAM MacBook
SAMPLE_SIZE = 20_000

# Maximum normal samples used for One-Class SVM
ONE_CLASS_NORMAL_LIMIT = 15_000


# ============================================================
# DATASET CONFIGURATION
#
# IMPORTANT:
#
# ToN-IoT:
#   X_train.joblib
#   y_train.joblib
#
# UNSW:
#   unsw_X_train.joblib
#   unsw_y_train.joblib
#
# CIC:
#   cic_X_train.joblib
#   cic_y_train.joblib
# ============================================================

DATASETS = {

    "UNSW-NB15": {
        "prefix": "unsw_",
    },

    "CIC-IDS": {
        "prefix": "cic_",
    },
}


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(
    dataset_name,
    prefix,
):

    print("\n" + "=" * 80)
    print(
        f"LOADING DATASET: {dataset_name}"
    )
    print("=" * 80)

    X_train = joblib.load(
        MODEL_DIR
        / f"{prefix}X_train.joblib"
    )

    y_train = joblib.load(
        MODEL_DIR
        / f"{prefix}y_train.joblib"
    )

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    return (
        X_train,
        y_train,
    )


# ============================================================
# CREATE STRATIFIED SAMPLE
# ============================================================

def create_stratified_sample(
    X,
    y,
    sample_size,
):

    actual_size = min(
        sample_size,
        X.shape[0],
    )

    if actual_size >= X.shape[0]:

        return X, y

    print(
        f"\nCreating stratified sample "
        f"of {actual_size:,} rows..."
    )

    X_sample, _, y_sample, _ = (
        train_test_split(
            X,
            y,
            train_size=actual_size,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    return (
        X_sample,
        y_sample,
    )


# ============================================================
# BINARY METRICS
# ============================================================

def calculate_binary_metrics(
    y_true,
    y_pred,
):

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


# ============================================================
# K-MEANS
# ============================================================

def run_kmeans(
    dataset_name,
    X,
    y,
):

    print("\n" + "-" * 80)
    print(
        f"{dataset_name} -> K-MEANS"
    )
    print("-" * 80)

    print(
        "Initializing K-Means..."
    )

    model = KMeans(
        n_clusters=2,
        n_init=10,
        random_state=RANDOM_STATE,
    )

    start = time.perf_counter()

    cluster_labels = (
        model.fit_predict(X)
    )

    training_time = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Cluster distribution
    # --------------------------------------------------------

    print(
        "\nCluster distribution:"
    )

    cluster_counts = (
        pd.Series(
            cluster_labels
        )
        .value_counts()
        .sort_index()
    )

    print(
        cluster_counts.to_string()
    )

    # --------------------------------------------------------
    # Map clusters to anomaly labels
    #
    # Each cluster is assigned based on the attack proportion
    # inside that cluster.
    # --------------------------------------------------------

    mapping = {}

    for cluster_id in np.unique(
        cluster_labels
    ):

        mask = (
            cluster_labels
            == cluster_id
        )

        attack_rate = np.mean(
            y[mask]
        )

        mapping[cluster_id] = (
            1
            if attack_rate >= 0.5
            else 0
        )

    y_pred = np.array(
        [
            mapping[cluster]
            for cluster in cluster_labels
        ]
    )

    metrics = calculate_binary_metrics(
        y,
        y_pred,
    )

    ari = adjusted_rand_score(
        y,
        cluster_labels,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\nAccuracy  : "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Precision : "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1 Score  : "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"ARI       : "
        f"{ari:.6f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y,
            y_pred,
        )
    )

    print(
        f"\nTraining time: "
        f"{training_time:.4f} sec"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    filename = (
        f"{dataset_name.lower()}"
        f"_kmeans.joblib"
    )

    model_path = (
        MODEL_DIR / filename
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"Model saved: {model_path}"
    )

    return {
        "dataset": dataset_name,
        "model": "K-Means",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "ari": ari,
        "training_time_sec": training_time,
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
    }


# ============================================================
# DBSCAN
# ============================================================

def run_dbscan(
    dataset_name,
    X,
    y,
):

    print("\n" + "-" * 80)
    print(
        f"{dataset_name} -> DBSCAN"
    )
    print("-" * 80)

    print(
        "Initializing DBSCAN..."
    )

    model = DBSCAN(
        eps=0.5,
        min_samples=10,
        n_jobs=-1,
    )

    start = time.perf_counter()

    labels = model.fit_predict(
        X
    )

    training_time = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Cluster distribution
    # --------------------------------------------------------

    print(
        "\nClusters found:"
    )

    unique_labels = np.unique(
        labels
    )

    for label in unique_labels:

        count = np.sum(
            labels == label
        )

        if label == -1:

            print(
                f"Noise (-1): "
                f"{count:,}"
            )

        else:

            print(
                f"Cluster {label}: "
                f"{count:,}"
            )

    # --------------------------------------------------------
    # Baseline:
    # DBSCAN noise = anomaly
    # --------------------------------------------------------

    y_pred = (
        labels == -1
    ).astype(int)

    metrics = calculate_binary_metrics(
        y,
        y_pred,
    )

    ari = adjusted_rand_score(
        y,
        labels,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\nAccuracy  : "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Precision : "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1 Score  : "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"ARI       : "
        f"{ari:.6f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y,
            y_pred,
        )
    )

    print(
        f"\nTraining time: "
        f"{training_time:.4f} sec"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    filename = (
        f"{dataset_name.lower()}"
        f"_dbscan.joblib"
    )

    model_path = (
        MODEL_DIR / filename
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"Model saved: {model_path}"
    )

    return {
        "dataset": dataset_name,
        "model": "DBSCAN",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "ari": ari,
        "training_time_sec": training_time,
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
    }


# ============================================================
# ONE-CLASS SVM
# ============================================================

def run_one_class_svm(
    dataset_name,
    X,
    y,
):

    print("\n" + "-" * 80)
    print(
        f"{dataset_name} -> ONE-CLASS SVM"
    )
    print("-" * 80)

    # --------------------------------------------------------
    # Only normal traffic is used for training
    # --------------------------------------------------------

    normal_mask = (
        y == 0
    )

    X_normal = X[
        normal_mask
    ]

    print(
        f"Normal samples available: "
        f"{X_normal.shape[0]:,}"
    )

    # --------------------------------------------------------
    # Limit normal samples for 8 GB RAM
    # --------------------------------------------------------

    if (
        X_normal.shape[0]
        > ONE_CLASS_NORMAL_LIMIT
    ):

        rng = np.random.default_rng(
            RANDOM_STATE
        )

        indices = rng.choice(
            X_normal.shape[0],
            size=ONE_CLASS_NORMAL_LIMIT,
            replace=False,
        )

        X_normal = X_normal[
            indices
        ]

    print(
        f"One-Class SVM training samples: "
        f"{X_normal.shape[0]:,}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.10,
    )

    print(
        "\nTraining One-Class SVM..."
    )

    start = time.perf_counter()

    model.fit(
        X_normal
    )

    training_time = (
        time.perf_counter()
        - start
    )

    print(
        f"Training completed in "
        f"{training_time:.4f} sec"
    )

    # --------------------------------------------------------
    # Prediction
    #
    # +1 = normal
    # -1 = anomaly
    # --------------------------------------------------------

    predictions = model.predict(
        X
    )

    y_pred = (
        predictions == -1
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_binary_metrics(
        y,
        y_pred,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\nAccuracy  : "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Precision : "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1 Score  : "
        f"{metrics['f1']:.6f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y,
            y_pred,
        )
    )

    print(
        f"\nTraining time: "
        f"{training_time:.4f} sec"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    filename = (
        f"{dataset_name.lower()}"
        f"_one_class_svm.joblib"
    )

    model_path = (
        MODEL_DIR / filename
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"Model saved: {model_path}"
    )

    return {
        "dataset": dataset_name,
        "model": "One-Class SVM",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "ari": np.nan,
        "training_time_sec": training_time,
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
    }


# ============================================================
# RUN ONE DATASET
# ============================================================

def run_dataset(
    dataset_name,
    prefix,
):

    X_train, y_train = load_dataset(
        dataset_name,
        prefix,
    )

    # --------------------------------------------------------
    # Common stratified sample
    # --------------------------------------------------------

    X_sample, y_sample = (
        create_stratified_sample(
            X_train,
            y_train,
            SAMPLE_SIZE,
        )
    )

    print("\n" + "=" * 80)
    print(
        f"{dataset_name} SAMPLE"
    )
    print("=" * 80)

    print(
        f"Samples : "
        f"{X_sample.shape[0]:,}"
    )

    print(
        f"Features: "
        f"{X_sample.shape[1]}"
    )

    print(
        "\nClass distribution:"
    )

    print(
        pd.Series(
            y_sample
        )
        .value_counts()
        .sort_index()
        .to_string()
    )

    results = []

    # --------------------------------------------------------
    # K-Means
    # --------------------------------------------------------

    results.append(
        run_kmeans(
            dataset_name,
            X_sample,
            y_sample,
        )
    )

    # --------------------------------------------------------
    # DBSCAN
    # --------------------------------------------------------

    results.append(
        run_dbscan(
            dataset_name,
            X_sample,
            y_sample,
        )
    )

    # --------------------------------------------------------
    # One-Class SVM
    # --------------------------------------------------------

    results.append(
        run_one_class_svm(
            dataset_name,
            X_sample,
            y_sample,
        )
    )

    return results


# ============================================================
# LOAD EXISTING ToN-IoT RESULTS
# ============================================================

def load_existing_ton_results():

    output_file = (
        RESULTS_DIR
        / "unsupervised_ml_results.csv"
    )

    if not output_file.exists():

        print(
            "\nWARNING:"
            "\nExisting ToN-IoT unsupervised "
            "results were not found."
        )

        return pd.DataFrame()

    existing_df = pd.read_csv(
        output_file
    )

    if "dataset" in existing_df.columns:

        ton_df = existing_df[
            existing_df["dataset"]
            == "ToN-IoT"
        ].copy()

    else:

        # Old ToN-IoT result file may not
        # contain dataset column.

        ton_df = existing_df.copy()

        ton_df.insert(
            0,
            "dataset",
            "ToN-IoT",
        )

    print(
        "\nExisting ToN-IoT results detected:"
    )

    print(
        ton_df[
            [
                "dataset",
                "model",
            ]
        ].to_string(
            index=False
        )
    )

    return ton_df


# ============================================================
# SAVE COMBINED RESULTS
# ============================================================

def save_results(
    new_results,
    existing_ton_df,
):

    output_file = (
        RESULTS_DIR
        / "unsupervised_ml_results.csv"
    )

    new_df = pd.DataFrame(
        new_results
    )

    # --------------------------------------------------------
    # Keep only ToN-IoT results from the existing file
    # --------------------------------------------------------

    frames = []

    if (
        existing_ton_df is not None
        and not existing_ton_df.empty
    ):

        frames.append(
            existing_ton_df
        )

    frames.append(
        new_df
    )

    results_df = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Normalize required columns
    # --------------------------------------------------------

    required_columns = [

        "dataset",
        "model",

        "accuracy",
        "precision",
        "recall",
        "f1",
        "ari",

        "training_time_sec",

        "tn",
        "fp",
        "fn",
        "tp",
    ]

    # Old ToN-IoT files may have
    # slightly different column names.

    rename_map = {

        "training_time_seconds":
            "training_time_sec",

    }

    for old, new in rename_map.items():

        if (
            old in results_df.columns
            and new not in results_df.columns
        ):

            results_df[new] = (
                results_df[old]
            )

    # --------------------------------------------------------
    # Add missing columns
    # --------------------------------------------------------

    for column in required_columns:

        if column not in results_df.columns:

            results_df[column] = np.nan

    results_df = results_df[
        required_columns
    ]

    # --------------------------------------------------------
    # Remove duplicate dataset/model pairs
    # --------------------------------------------------------

    results_df = (
        results_df
        .drop_duplicates(
            subset=[
                "dataset",
                "model",
            ],
            keep="last",
        )
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
        f"\nCombined results saved to:"
        f"\n{output_file}"
    )

    return results_df


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results_df
):

    print("\n" + "=" * 120)

    print(
        "UNSUPERVISED / "
        "SEMI-SUPERVISED RESULTS"
    )

    print("=" * 120)

    columns = [

        "dataset",
        "model",

        "accuracy",
        "precision",
        "recall",
        "f1",
        "ari",

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
# MAIN
# ============================================================

def main():

    print("\n" + "#" * 90)

    print(
        "UNSUPERVISED / "
        "SEMI-SUPERVISED "
        "TRAFFIC ANOMALY DETECTION"
    )

    print("#" * 90)

    print(
        f"\nSample size: "
        f"{SAMPLE_SIZE:,} per dataset"
    )

    print(
        "One-Class SVM normal training limit: "
        f"{ONE_CLASS_NORMAL_LIMIT:,}"
    )

    # --------------------------------------------------------
    # Preserve existing ToN-IoT results
    # --------------------------------------------------------

    existing_ton_df = (
        load_existing_ton_results()
    )

    # --------------------------------------------------------
    # Run UNSW and CIC
    # --------------------------------------------------------

    all_results = []

    for dataset_name, config in (
        DATASETS.items()
    ):

        print("\n\n")

        print(
            "#" * 90
        )

        print(
            f"PROCESSING: "
            f"{dataset_name}"
        )

        print(
            "#" * 90
        )

        results = run_dataset(
            dataset_name,
            config["prefix"],
        )

        all_results.extend(
            results
        )

    # --------------------------------------------------------
    # Combine with existing ToN results
    # --------------------------------------------------------

    results_df = save_results(
        all_results,
        existing_ton_df,
    )

    # --------------------------------------------------------
    # Final table
    # --------------------------------------------------------

    print_results(
        results_df
    )

    print("\n" + "#" * 90)

    print(
        "UNSUPERVISED EXPERIMENTS COMPLETE"
    )

    print("#" * 90)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()