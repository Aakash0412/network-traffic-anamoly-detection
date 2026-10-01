from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    adjusted_rand_score,
    roc_auc_score,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

# Keep unsupervised experiments computationally manageable
UNSUPERVISED_SAMPLE_SIZE = 20000


# ============================================================
# Load Data
# ============================================================

print("Loading preprocessed data...")

X_train = joblib.load(MODEL_DIR / "X_train.joblib")
y_train = joblib.load(MODEL_DIR / "y_train.joblib")

X_test = joblib.load(MODEL_DIR / "X_test.joblib")
y_test = joblib.load(MODEL_DIR / "y_test.joblib")

print(f"Training data : {X_train.shape}")
print(f"Test data     : {X_test.shape}")


# ============================================================
# Create Sample
# ============================================================

sample_size = min(
    UNSUPERVISED_SAMPLE_SIZE,
    X_train.shape[0]
)

rng = np.random.RandomState(RANDOM_STATE)

sample_indices = rng.choice(
    X_train.shape[0],
    size=sample_size,
    replace=False
)

X_sample = X_train[sample_indices]
y_sample = np.asarray(y_train)[sample_indices]

print("\nUnsupervised sample:")
print(f"Samples: {X_sample.shape[0]}")
print(f"Features: {X_sample.shape[1]}")


# ============================================================
# K-MEANS
# ============================================================

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING")
print("=" * 70)

start_time = time.time()

kmeans = KMeans(
    n_clusters=2,
    random_state=RANDOM_STATE,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_sample)

kmeans_time = time.time() - start_time

print(f"Training time: {kmeans_time:.2f} seconds")

print("\nCluster distribution:")
print(pd.Series(kmeans_labels).value_counts().sort_index())

ari = adjusted_rand_score(
    y_sample,
    kmeans_labels
)

print(f"\nAdjusted Rand Index: {ari:.4f}")

joblib.dump(
    kmeans,
    MODEL_DIR / "kmeans.joblib"
)

print("Model saved: models/kmeans.joblib")


# ============================================================
# Map K-Means Clusters to Binary Labels
# ============================================================

cluster_mapping = {}

for cluster in np.unique(kmeans_labels):

    mask = kmeans_labels == cluster

    majority_label = int(
        np.round(y_sample[mask].mean())
    )

    cluster_mapping[cluster] = majority_label

kmeans_predictions = np.array([
    cluster_mapping[c]
    for c in kmeans_labels
])

kmeans_accuracy = accuracy_score(
    y_sample,
    kmeans_predictions
)

kmeans_precision = precision_score(
    y_sample,
    kmeans_predictions,
    zero_division=0
)

kmeans_recall = recall_score(
    y_sample,
    kmeans_predictions,
    zero_division=0
)

kmeans_f1 = f1_score(
    y_sample,
    kmeans_predictions,
    zero_division=0
)

print("\nK-Means Binary Evaluation:")
print(f"Accuracy  : {kmeans_accuracy:.4f}")
print(f"Precision : {kmeans_precision:.4f}")
print(f"Recall    : {kmeans_recall:.4f}")
print(f"F1 Score  : {kmeans_f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_sample,
    kmeans_predictions
))


# ============================================================
# DBSCAN
# ============================================================

print("\n" + "=" * 70)
print("DBSCAN")
print("=" * 70)

start_time = time.time()

dbscan = DBSCAN(
    eps=2.0,
    min_samples=10,
    n_jobs=-1
)

dbscan_labels = dbscan.fit_predict(X_sample)

dbscan_time = time.time() - start_time

print(f"Training time: {dbscan_time:.2f} seconds")

unique_labels, counts = np.unique(
    dbscan_labels,
    return_counts=True
)

print("\nCluster distribution:")

for label, count in zip(unique_labels, counts):
    if label == -1:
        print(f"Noise (-1): {count}")
    else:
        print(f"Cluster {label}: {count}")

joblib.dump(
    dbscan,
    MODEL_DIR / "dbscan.joblib"
)

print("Model saved: models/dbscan.joblib")


# ============================================================
# DBSCAN Evaluation
# ============================================================

# DBSCAN does not directly produce binary attack labels.
# Noise points (-1) are treated as potential anomalies.

dbscan_predictions = (
    dbscan_labels == -1
).astype(int)

dbscan_accuracy = accuracy_score(
    y_sample,
    dbscan_predictions
)

dbscan_precision = precision_score(
    y_sample,
    dbscan_predictions,
    zero_division=0
)

dbscan_recall = recall_score(
    y_sample,
    dbscan_predictions,
    zero_division=0
)

dbscan_f1 = f1_score(
    y_sample,
    dbscan_predictions,
    zero_division=0
)

print("\nDBSCAN Anomaly Evaluation:")
print(f"Accuracy  : {dbscan_accuracy:.4f}")
print(f"Precision : {dbscan_precision:.4f}")
print(f"Recall    : {dbscan_recall:.4f}")
print(f"F1 Score  : {dbscan_f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_sample,
    dbscan_predictions
))


# ============================================================
# ONE-CLASS SVM
# ============================================================

print("\n" + "=" * 70)
print("ONE-CLASS SVM")
print("=" * 70)

start_time = time.time()

one_class_svm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.10
)

one_class_svm.fit(X_sample)

one_class_time = time.time() - start_time

print(f"Training time: {one_class_time:.2f} seconds")

# One-Class SVM:
# +1 = normal
# -1 = anomaly

ocsvm_raw = one_class_svm.predict(X_sample)

ocsvm_predictions = (
    ocsvm_raw == -1
).astype(int)

ocsvm_accuracy = accuracy_score(
    y_sample,
    ocsvm_predictions
)

ocsvm_precision = precision_score(
    y_sample,
    ocsvm_predictions,
    zero_division=0
)

ocsvm_recall = recall_score(
    y_sample,
    ocsvm_predictions,
    zero_division=0
)

ocsvm_f1 = f1_score(
    y_sample,
    ocsvm_predictions,
    zero_division=0
)

print("\nOne-Class SVM Evaluation:")
print(f"Accuracy  : {ocsvm_accuracy:.4f}")
print(f"Precision : {ocsvm_precision:.4f}")
print(f"Recall    : {ocsvm_recall:.4f}")
print(f"F1 Score  : {ocsvm_f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_sample,
    ocsvm_predictions
))

joblib.dump(
    one_class_svm,
    MODEL_DIR / "one_class_svm.joblib"
)

print("Model saved: models/one_class_svm.joblib")


# ============================================================
# Save Results
# ============================================================

results = pd.DataFrame([
    {
        "model": "K-Means",
        "accuracy": kmeans_accuracy,
        "precision": kmeans_precision,
        "recall": kmeans_recall,
        "f1": kmeans_f1,
        "training_time_seconds": kmeans_time,
    },
    {
        "model": "DBSCAN",
        "accuracy": dbscan_accuracy,
        "precision": dbscan_precision,
        "recall": dbscan_recall,
        "f1": dbscan_f1,
        "training_time_seconds": dbscan_time,
    },
    {
        "model": "One-Class SVM",
        "accuracy": ocsvm_accuracy,
        "precision": ocsvm_precision,
        "recall": ocsvm_recall,
        "f1": ocsvm_f1,
        "training_time_seconds": one_class_time,
    },
])

results_path = (
    RESULTS_DIR /
    "unsupervised_ml_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print("\n" + "=" * 70)
print("UNSUPERVISED MODEL FILE READY")
print("=" * 70)

print(results.to_string(index=False))

print(f"\nResults saved to: {results_path}")