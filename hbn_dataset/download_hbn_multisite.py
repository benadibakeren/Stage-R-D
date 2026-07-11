"""
Download HBN-EEG subjects from multiple release sites for multi-site FL simulation.
Downloads only RestingState EEG files to minimize storage usage.

Sites used:
- ds005515: 30 subjects (15 ADHD + 15 Control)
- ds005511: 30 subjects (15 ADHD + 15 Control)
- ds005509: 30 subjects (15 ADHD + 15 Control)

Author: Keren Benadiba
Institution: UMONS, Service SEMi
"""

import openneuro as on
import os
import pandas as pd
import numpy as np

RELEASES = {
    "ds005515": "hbn_tsv/participants_ds005515.tsv",
    "ds005511": "hbn_tsv/participants_ds005511.tsv",
    "ds005509": "hbn_tsv/participants_ds005509.tsv",
}

N_PER_CLASS  = 15
OUTPUT_DIR   = "eeg_data/hbn_multisite"

def select_subjects(tsv_path, n_per_class=15, seed=42):
    """Select balanced ADHD/Control subjects with RestingState available."""
    df = pd.read_csv(tsv_path, sep="\t")
    if "RestingState" in df.columns:
        df = df[df["RestingState"] == "available"]
    adhd    = df[df["attention"] > 0.5]["participant_id"].tolist()
    control = df[df["attention"] < -0.5]["participant_id"].tolist()
    np.random.seed(seed)
    selected = (
        np.random.choice(adhd, min(n_per_class, len(adhd)), replace=False).tolist() +
        np.random.choice(control, min(n_per_class, len(control)), replace=False).tolist()
    )
    return selected

if __name__ == "__main__":
    for release, tsv_path in RELEASES.items():
        print(f"\nDownloading {release}...")
        subjects   = select_subjects(tsv_path, N_PER_CLASS)
        target_dir = f"{OUTPUT_DIR}/{release}"
        os.makedirs(target_dir, exist_ok=True)

        include_files = []
        for subject in subjects:
            include_files.append(f"{subject}/eeg/{subject}_task-RestingState_eeg.set")
            include_files.append(f"{subject}/eeg/{subject}_task-RestingState_channels.tsv")

        try:
            on.download(
                dataset=release,
                target_dir=target_dir,
                include=include_files
            )
            print(f"✅ {release} downloaded — {len(subjects)} subjects")
        except Exception as e:
            print(f"❌ Error {release}: {e}")