"""
diagnostics/serialization_ablation.py

Diagnostic experiment: does the text serialization format given to
BiomedBERT (numeric floats vs. narrative clinical phrasing) affect
classification accuracy on Nasrabadi, once the Patient-ID leakage is
corrected and hyperparameters are held identical?

Two formats are compared, at strictly identical hyperparameters
(5 epochs, lr=3e-5, batch=16, standard CrossEntropyLoss, no class
weighting), evaluated via patient-level 5-fold GroupKFold on all 121
patients:

  - numeric   : "Frontal Theta/Beta Ratio (TBR): 7.19"
  - narrative : "Electroencephalographic evaluation reveals a significantly
                 elevated frontal theta/beta ratio..."

Categorical bucketing ("TBR: ELEVATED") was also explored during
development and collapsed identically to the numeric baseline on a single
split; it is included here as an optional third condition but is not run
by default, to keep this script's default output matching the numbers
reported in README.md.

Threshold values used to build the categorical/narrative phrases are
recomputed independently inside each fold, using only that fold's
training patients, to avoid any information leakage between folds.

Usage:
    python diagnostics/serialization_ablation.py --data data/patient_features.csv

Expected columns in the CSV: ID, label, TBR, FAA, theta_mean, beta_mean, alpha_mean
(label: 0 = Control, 1 = ADHD)

Note: this script trains 10 full BiomedBERT models (5 folds x 2 formats).
Expect this to take substantially longer than a single training run.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


# ==========================================================================
# Text serialization functions
# ==========================================================================
def make_text_numeric(row) -> str:
    return (
        f"Patient ID: PATIENT_ANONYMIZED\n"
        f"Frontal Theta/Beta Ratio (TBR): {row['TBR']:.3f}\n"
        f"Frontal Alpha Asymmetry (FAA): {row['FAA']:.3f}\n"
        f"Theta Power: {row['theta_mean']:.3f}\n"
        f"Beta Power: {row['beta_mean']:.3f}"
    )


def make_text_categorical(row, q: dict) -> str:
    def bucket(v, lo, hi):
        if v < lo:
            return "LOW"
        elif v > hi:
            return "ELEVATED"
        return "MODERATE"

    return (
        f"Patient ID: PATIENT_ANONYMIZED\n"
        f"Frontal Theta/Beta Ratio (TBR): {bucket(row['TBR'], q['tbr_q1'], q['tbr_q2'])}\n"
        f"Frontal Alpha Asymmetry (FAA): {bucket(row['FAA'], q['faa_q1'], q['faa_q2'])}\n"
        f"Theta Power: {bucket(row['theta_mean'], q['theta_q1'], q['theta_q2'])}\n"
        f"Beta Power: {bucket(row['beta_mean'], q['beta_q1'], q['beta_q2'])}"
    )


def make_clinical_narrative(row, q: dict) -> str:
    def tbr_phrase(v):
        if v > q["tbr_q2"]:
            return "a significantly elevated frontal theta/beta ratio"
        elif v < q["tbr_q1"]:
            return "a markedly reduced frontal theta/beta ratio"
        return "a frontal theta/beta ratio within the typical range"

    def faa_phrase(v):
        side = "left-dominant" if v < 0 else "right-dominant"
        sign = "negative" if v < 0 else "positive"
        return f"characterized as {sign} ({side})"

    def theta_phrase(v):
        if v > q["theta_q2"]:
            return "prominent theta activity"
        elif v < q["theta_q1"]:
            return "reduced theta activity"
        return "typical theta activity"

    def beta_phrase(v):
        if v > q["beta_q2"]:
            return "elevated beta wave dynamics"
        elif v < q["beta_q1"]:
            return "diminished beta wave dynamics"
        return "standard beta wave dynamics"

    return (
        f"Electroencephalographic evaluation reveals {tbr_phrase(row['TBR'])}. "
        f"Frontal alpha asymmetry is {faa_phrase(row['FAA'])}. "
        f"Spectral power analysis highlights {theta_phrase(row['theta_mean'])} "
        f"accompanied by {beta_phrase(row['beta_mean'])}."
    )


def compute_fold_thresholds(train_df: pd.DataFrame) -> dict:
    """Quantile thresholds computed ONLY on the training patients of a given fold."""
    return {
        "tbr_q1": train_df["TBR"].quantile(0.33), "tbr_q2": train_df["TBR"].quantile(0.66),
        "faa_q1": train_df["FAA"].quantile(0.33), "faa_q2": train_df["FAA"].quantile(0.66),
        "theta_q1": train_df["theta_mean"].quantile(0.33), "theta_q2": train_df["theta_mean"].quantile(0.66),
        "beta_q1": train_df["beta_mean"].quantile(0.33), "beta_q2": train_df["beta_mean"].quantile(0.66),
    }


# ==========================================================================
# Dataset + training
# ==========================================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


def train_and_predict(train_df, test_df, text_column, fold_id, run_name):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_dataset = TextDataset(train_df[text_column], train_df["label"], tokenizer)
    test_dataset = TextDataset(test_df[text_column], test_df["label"], tokenizer)

    training_args = TrainingArguments(
        output_dir=f"./results_{run_name}_fold{fold_id}",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        # Standard CrossEntropyLoss, no class weighting — kept identical
        # across every condition tested in this script on purpose.
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, compute_metrics=compute_metrics,
    )
    trainer.train()

    preds = np.argmax(trainer.predict(test_dataset).predictions, axis=1)
    return preds


# ==========================================================================
# Main: 5-fold patient-level GroupKFold, for each serialization format
# ==========================================================================
def main(data_path: str, conditions=("numeric", "narrative")):
    df = pd.read_csv(data_path).reset_index(drop=True)
    print(f"Total patients: {len(df)} — {df['label'].value_counts().to_dict()}")

    gkf = GroupKFold(n_splits=5)
    groups = df["ID"]

    results = {
        cond: {"fold_acc": [], "oof_preds": np.full(len(df), -1), "oof_true": df["label"].values}
        for cond in conditions
    }

    for fold_id, (train_idx, test_idx) in enumerate(gkf.split(df, df["label"], groups), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        q = compute_fold_thresholds(train_df)  # train-only, per fold

        train_df["numeric"] = train_df.apply(make_text_numeric, axis=1)
        test_df["numeric"] = test_df.apply(make_text_numeric, axis=1)

        train_df["categorical"] = train_df.apply(lambda r: make_text_categorical(r, q), axis=1)
        test_df["categorical"] = test_df.apply(lambda r: make_text_categorical(r, q), axis=1)

        train_df["narrative"] = train_df.apply(lambda r: make_clinical_narrative(r, q), axis=1)
        test_df["narrative"] = test_df.apply(lambda r: make_clinical_narrative(r, q), axis=1)

        print(f"\n{'#' * 60}\nFOLD {fold_id}/5 — train={len(train_df)}, test={len(test_df)}\n{'#' * 60}")

        for cond in conditions:
            preds = train_and_predict(train_df, test_df, cond, fold_id, cond)
            acc = accuracy_score(test_df["label"], preds)
            results[cond]["fold_acc"].append(acc)
            results[cond]["oof_preds"][test_idx] = preds
            print(f"  [{cond}] Fold {fold_id} accuracy: {acc * 100:.2f}%")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for cond in conditions:
        fold_acc = np.array(results[cond]["fold_acc"])
        oof_acc = accuracy_score(results[cond]["oof_true"], results[cond]["oof_preds"])
        cm = confusion_matrix(results[cond]["oof_true"], results[cond]["oof_preds"])
        print(f"\n[{cond.upper()}]")
        print("  Fold accuracies:", [f"{a * 100:.2f}%" for a in fold_acc])
        print(f"  Mean ± std: {fold_acc.mean() * 100:.2f}% ± {fold_acc.std() * 100:.2f}%")
        print(f"  Out-of-fold accuracy (121 patients): {oof_acc * 100:.2f}%")
        print("  Out-of-fold confusion matrix:")
        print(" ", cm)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=str, default="data/patient_features.csv",
        help="Path to the CSV file with patient-level tabular EEG features.",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=["numeric", "narrative"],
        choices=["numeric", "categorical", "narrative"],
        help="Which serialization formats to compare (default: numeric narrative).",
    )
    args = parser.parse_args()
    main(args.data, tuple(args.conditions))