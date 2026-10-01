from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def inspect_csv(path, name, nrows=3):
    print("\n" + "=" * 90)
    print(f"DATASET: {name}")
    print(f"FILE: {path}")
    print("=" * 90)

    # Read only a few rows first
    df = pd.read_csv(path, nrows=nrows, low_memory=False)

    print(f"\nNumber of columns: {len(df.columns)}")

    print("\nColumns:")
    for i, col in enumerate(df.columns, start=1):
        print(f"{i:3}. {repr(col)}")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst rows:")
    print(df.head(nrows).to_string())

    # Possible target columns
    target_candidates = [
        col for col in df.columns
        if any(
            keyword in col.lower()
            for keyword in [
                "label",
                "attack",
                "class",
                "target",
                "category",
                "anomaly"
            ]
        )
    ]

    print("\nPossible target columns:")
    print(target_candidates)


# -------------------------------------------------------------------
# 1. ToN-IoT
# -------------------------------------------------------------------

ton_iot = DATA_DIR / "train_test_network.csv"

inspect_csv(
    ton_iot,
    "ToN-IoT"
)


# -------------------------------------------------------------------
# 2. UNSW-NB15
# -------------------------------------------------------------------

unsw_train = DATA_DIR / "archive" / "UNSW_NB15_training-set.csv"
unsw_test = DATA_DIR / "archive" / "UNSW_NB15_testing-set.csv"

inspect_csv(
    unsw_train,
    "UNSW-NB15 Training Set"
)

inspect_csv(
    unsw_test,
    "UNSW-NB15 Testing Set"
)


# -------------------------------------------------------------------
# 3. CIC-IDS
# -------------------------------------------------------------------

cic_dir = DATA_DIR / "archive-2"

cic_files = sorted(cic_dir.glob("*.csv"))

print("\n\n" + "#" * 90)
print("CIC-IDS FILES")
print("#" * 90)

for file in cic_files:
    print(f"\nFile: {file.name}")

    df = pd.read_csv(
        file,
        nrows=3,
        low_memory=False
    )

    print(f"Columns: {len(df.columns)}")

    print("Possible target columns:")
    targets = [
        col for col in df.columns
        if any(
            keyword in col.lower()
            for keyword in [
                "label",
                "attack",
                "class",
                "target",
                "category",
                "anomaly"
            ]
        )
    ]

    print(targets)

    print("First few column names:")
    print([repr(col) for col in df.columns[:10]])

    print("Last few column names:")
    print([repr(col) for col in df.columns[-10:]])