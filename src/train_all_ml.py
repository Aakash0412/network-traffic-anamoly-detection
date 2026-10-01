from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)


from supervised_ml import main as run_supervised
from unsupervised_ml import main as run_unsupervised


def main():

    print("\n")
    print("#" * 100)
    print("NETWORK TRAFFIC ANOMALY DETECTION")
    print("CLASSICAL MACHINE LEARNING PIPELINE")
    print("#" * 100)

    # ========================================================
    # SUPERVISED
    # ========================================================

    print("\n\n")
    print("#" * 100)
    print("PART 1: SUPERVISED MACHINE LEARNING")
    print("#" * 100)

    run_supervised()

    # ========================================================
    # UNSUPERVISED
    # ========================================================

    print("\n\n")
    print("#" * 100)
    print("PART 2: UNSUPERVISED / ANOMALY DETECTION")
    print("#" * 100)

    run_unsupervised()

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n\n")
    print("#" * 100)
    print("ALL CLASSICAL ML EXPERIMENTS COMPLETE")
    print("#" * 100)

    print("\nResults available under:")

    print(
        BASE_DIR /
        "results" /
        "metrics"
    )


if __name__ == "__main__":
    main()