"""
diagnostics/random_forest_baseline.py

Diagnostic experiment: does a classical model, trained directly on the raw
tabular EEG biomarkers (no text serialization), find a learnable ADHD signal
on the Nasrabadi dataset?

This was run after BiomedBERT collapsed to a constant majority-class
prediction (44.26% patient-level accuracy) once the Patient-ID leakage was
corrected (see README.md, Step 2). The goal is to isolate whether the null
result comes from an absence of signal in the data, or from the text
serialization pipeline used to feed BiomedBERT.

Usage:
    python diagnostics/random_forest_baseline.py --data data/patient_features.csv

Expected columns in the CSV: ID, label, TBR, FAA, theta_mean, beta_mean, alpha_mean
(label: 0 = Control, 1 = ADHD)
"""

import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score


def main(data_path: str, random_state: int = 42):
    df = pd.read_csv(data_path)
    feature_cols = ["TBR", "FAA", "theta_mean", "beta_mean", "alpha_mean"]
    X = df[feature_cols]
    y = df["label"]

    print(f"Loaded {len(df)} patients — label distribution: {y.value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # Part 1 — single representative split (illustrative confusion matrix)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    single_acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print(f"Single split accuracy: {single_acc * 100:.2f}%")
    print("Confusion matrix (rows = true, cols = predicted, [Control, ADHD]):")
    print(cm)
    print("=" * 50)

    # ------------------------------------------------------------------
    # Part 2 — 5-fold stratified cross-validation (the robust headline number)
    # ------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        RandomForestClassifier(n_estimators=200, random_state=random_state),
        X, y, cv=skf,
    )

    print("\n" + "=" * 50)
    print("5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 50)
    print("Fold scores:", [f"{s * 100:.2f}%" for s in scores])
    print(f"Mean accuracy: {scores.mean() * 100:.2f}% ± {scores.std() * 100:.2f}%")
    print("=" * 50)

    print(
        "\nInterpretation: a score consistently above chance (50%), using both "
        "classes (see confusion matrix above), indicates the frontal EEG "
        "biomarkers carry a real, modest discriminative signal at this sample "
        "size — independently of BiomedBERT's text-serialization pipeline."
    )

    return {
        "single_split_accuracy": single_acc,
        "single_split_confusion_matrix": cm,
        "cv_scores": scores,
        "cv_mean": scores.mean(),
        "cv_std": scores.std(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=str, default="data/patient_features.csv",
        help="Path to the CSV file with patient-level tabular EEG features.",
    )
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(args.data, args.random_state)