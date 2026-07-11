"""
Multi-Site Federated Learning Simulation on HBN-EEG
=====================================================
Simulates FL across 3 genuine HBN-EEG collection sites.

Each site = 1 independent hospital node.
FedAvg aggregation on CPU to avoid GPU memory constraints.
Patient-level evaluation via majority voting.

Key results:
- Single-site (ds005515): 38.89%
- Centralized (3 sites) : 61.11%
- Best FL (Round 3)     : 61.11%
- Gain FL vs single-site: +22.22%

Author: Keren Benadiba
Institution: UMONS, Service SEMi
"""

import torch
import copy
import gc
import numpy as np
import pandas as pd
import os
from collections import Counter
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from scipy.signal import welch
import mne

# ── Configuration ─────────────────────────────────────────
MODEL_NAME   = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
DATA_DIR     = "eeg_data/hbn_multisite"
HBN_TSV_DIR  = "hbn_tsv"
NUM_ROUNDS   = 5
LOCAL_EPOCHS = 5
LEARNING_RATE = 3e-5
BATCH_SIZE   = 4
SEED         = 42

RELEASES = {
    "ds005515": f"{HBN_TSV_DIR}/participants_ds005515.tsv",
    "ds005511": f"{HBN_TSV_DIR}/participants_ds005511.tsv",
    "ds005509": f"{HBN_TSV_DIR}/participants_ds005509.tsv",
}

# ── Setup ─────────────────────────────────────────────────
np.random.seed(SEED)
torch.manual_seed(SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── EEG Feature Extraction ────────────────────────────────
def extract_features_hbn(subject_id, release):
    """Extract TBR, FAA and spectral power from HBN-EEG RestingState."""
    eeg_file = f"{DATA_DIR}/{release}/{subject_id}/eeg/{subject_id}_task-RestingState_eeg.set"
    try:
        raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
        frontal_gsn = {'Fz':'E11', 'F3':'E24', 'F4':'E124', 'Fp1':'E22', 'Fp2':'E9'}
        frontal   = list(frontal_gsn.values())
        available = [ch for ch in frontal if ch in raw.ch_names]
        if len(available) == 0:
            return None
        raw_frontal = raw.copy().pick(available)
        data  = raw_frontal.get_data()
        sfreq = raw.info['sfreq']
        theta_p, beta_p, alpha_p = [], [], []
        for ch_data in data:
            freqs, psd = welch(ch_data, fs=sfreq, nperseg=int(sfreq*2))
            theta_p.append(np.trapezoid(psd[(freqs>=4)&(freqs<=8)],   freqs[(freqs>=4)&(freqs<=8)]))
            beta_p.append(np.trapezoid(psd[(freqs>=13)&(freqs<=30)],  freqs[(freqs>=13)&(freqs<=30)]))
            alpha_p.append(np.trapezoid(psd[(freqs>=8)&(freqs<=13)],  freqs[(freqs>=8)&(freqs<=13)]))
        tbr = np.mean(theta_p) / np.mean(beta_p)
        faa = alpha_p[available.index('E124')] - alpha_p[available.index('E24')] \
              if 'E24' in available and 'E124' in available else 0.0
        return {
            'ID'        : subject_id,
            'release'   : release,
            'TBR'       : tbr,
            'FAA'       : faa,
            'theta_mean': np.mean(theta_p),
            'beta_mean' : np.mean(beta_p),
            'alpha_mean': np.mean(alpha_p)
        }
    except Exception as e:
        print(f"  ❌ {subject_id}: {e}")
        return None

def generate_text(row):
    """Serialize EEG features into structured clinical text."""
    return f"""Patient ID: {row['ID']}
Site: {row['release']}
Frontal Theta/Beta Ratio (TBR): {row['TBR']:.3f}
Frontal Alpha Asymmetry (FAA): {row['FAA']:.6f}
Frontal Theta Power: {row['theta_mean']:.6e} V2/Hz
Frontal Beta Power: {row['beta_mean']:.6e} V2/Hz
Frontal Alpha Power: {row['alpha_mean']:.6e} V2/Hz"""

# ── Dataset ───────────────────────────────────────────────
class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {
            "input_ids"     : self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label"         : torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── Patient-level evaluation ──────────────────────────────
def evaluate_patient_level(model, eval_df):
    """Evaluate model at patient level using majority voting."""
    model.eval()
    texts  = eval_df["text"].tolist()
    labels = eval_df["label"].tolist()
    ids    = eval_df["patient_id"].tolist()
    loader = DataLoader(EEGDataset(texts, labels), batch_size=BATCH_SIZE)
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids      =batch["input_ids"].to(device),
                attention_mask =batch["attention_mask"].to(device)
            )
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
    patient_preds  = {}
    patient_labels = {}
    for pid, pred, label in zip(ids, all_preds, labels):
        if pid not in patient_preds:
            patient_preds[pid]  = []
            patient_labels[pid] = label
        patient_preds[pid].append(pred)
    final_preds  = [Counter(v).most_common(1)[0][0] for v in patient_preds.values()]
    final_labels = list(patient_labels.values())
    return accuracy_score(final_labels, final_preds), len(final_labels)

# ── Local training ────────────────────────────────────────
def train_local(model, train_loader, epochs=LOCAL_EPOCHS, lr=LEARNING_RATE):
    """Train model locally on a single node."""
    model.train()
    optimizer    = AdamW(model.parameters(), lr=lr)
    scaler       = torch.amp.GradScaler('cuda')
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids      =batch["input_ids"].to(device),
                    attention_mask =batch["attention_mask"].to(device),
                    labels         =batch["label"].to(device)
                )
            scaler.scale(outputs.loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += outputs.loss.item()
        print(f"    Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
    return model

# ── FedAvg on CPU ─────────────────────────────────────────
def fed_avg_cpu(global_model, local_state_dicts, node_sizes):
    """FedAvg aggregation on CPU to avoid GPU memory constraints."""
    global_state = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
    total = sum(node_sizes)
    for key in global_state:
        global_state[key] = sum(
            local_state_dicts[i][key] * (node_sizes[i] / total)
            for i in range(len(local_state_dicts))
        )
    global_model.load_state_dict(global_state)
    return global_model

# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":

    # Load and serialize data
    all_records = []
    for release, tsv_path in RELEASES.items():
        df_p = pd.read_csv(tsv_path, sep="\t")
        target_dir = f"{DATA_DIR}/{release}"
        if not os.path.exists(target_dir):
            print(f"❌ {release} not found — run download_hbn_multisite.py first")
            continue
        print(f"\nExtracting {release}...")
        for subject in os.listdir(target_dir):
            if not subject.startswith("sub-"):
                continue
            features = extract_features_hbn(subject, release)
            if features:
                row = df_p[df_p["participant_id"] == subject]
                if len(row) > 0:
                    att = row["attention"].values[0]
                    if att > 0.5:   features["label"] = 1
                    elif att < -0.5: features["label"] = 0
                    else: continue
                    all_records.append(features)

    df_hbn = pd.DataFrame(all_records)
    df_hbn["text"]       = df_hbn.apply(generate_text, axis=1)
    df_hbn["patient_id"] = df_hbn["ID"]
    print(f"\n✅ {len(df_hbn)} subjects extracted")

    # Fixed train/test split
    np.random.seed(SEED)
    all_patients = df_hbn["patient_id"].unique()
    np.random.shuffle(all_patients)
    split_idx      = int(0.8 * len(all_patients))
    train_patients = all_patients[:split_idx]
    test_patients  = all_patients[split_idx:]
    train_df = df_hbn[df_hbn["patient_id"].isin(train_patients)].reset_index(drop=True)
    test_df  = df_hbn[df_hbn["patient_id"].isin(test_patients)].reset_index(drop=True)
    print(f"Train: {train_df['patient_id'].nunique()} patients")
    print(f"Test : {test_df['patient_id'].nunique()} patients")

    # Baseline single-site
    print("\n" + "="*50)
    print("BASELINE SINGLE-SITE (ds005515)")
    print("="*50)
    single_site_df = train_df[train_df["release"] == "ds005515"]
    single_model   = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)
    train_loader_single = DataLoader(
        EEGDataset(single_site_df["text"].tolist(), single_site_df["label"].tolist()),
        batch_size=BATCH_SIZE, shuffle=True
    )
    single_model = train_local(single_model, train_loader_single)
    single_acc, n_p = evaluate_patient_level(single_model, test_df)
    print(f"\nBaseline single-site: {single_acc*100:.2f}% ({n_p} patients)")
    del single_model
    torch.cuda.empty_cache()
    gc.collect()

    # Centralized baseline
    print("\n" + "="*50)
    print("CENTRALIZED BASELINE (3 sites)")
    print("="*50)
    central_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)
    train_loader_central = DataLoader(
        EEGDataset(train_df["text"].tolist(), train_df["label"].tolist()),
        batch_size=BATCH_SIZE, shuffle=True
    )
    central_model = train_local(central_model, train_loader_central)
    central_acc, n_p = evaluate_patient_level(central_model, test_df)
    print(f"\nCentralized baseline: {central_acc*100:.2f}% ({n_p} patients)")
    del central_model
    torch.cuda.empty_cache()
    gc.collect()

    # FL multi-site
    print("\n" + "="*50)
    print("FEDERATED LEARNING - 3 HBN sites")
    print("="*50)
    node_releases    = list(RELEASES.keys())
    global_model_cpu = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    best_fl_acc   = 0
    round_results = []

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n--- Round {round_num}/{NUM_ROUNDS} ---")
        local_state_dicts = []
        node_sizes        = []

        for release in node_releases:
            node_df = train_df[train_df["release"] == release]
            if len(node_df) == 0:
                continue
            local_model = copy.deepcopy(global_model_cpu).to(device)
            train_loader = DataLoader(
                EEGDataset(node_df["text"].tolist(), node_df["label"].tolist()),
                batch_size=BATCH_SIZE, shuffle=True
            )
            print(f"\n  Site {release} ({node_df['patient_id'].nunique()} patients)...")
            local_model = train_local(local_model, train_loader)
            local_state_dicts.append(
                {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
            )
            node_sizes.append(node_df["patient_id"].nunique())
            del local_model
            torch.cuda.empty_cache()
            gc.collect()

        global_model_cpu = fed_avg_cpu(global_model_cpu, local_state_dicts, node_sizes)
        del local_state_dicts
        gc.collect()

        eval_model = copy.deepcopy(global_model_cpu).to(device)
        fl_acc, n_p = evaluate_patient_level(eval_model, test_df)
        del eval_model
        torch.cuda.empty_cache()
        gc.collect()

        round_results.append(fl_acc)
        if fl_acc > best_fl_acc:
            best_fl_acc = fl_acc
        print(f"\n  Round {round_num}: {fl_acc*100:.2f}% ({n_p} patients)")

    # Final summary
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS - MULTI-SITE FL HBN")
    print(f"{'='*50}")
    print(f"Single-site baseline (ds005515) : {single_acc*100:.2f}%")
    print(f"Centralized baseline (3 sites)  : {central_acc*100:.2f}%")
    print(f"Best FL accuracy                : {best_fl_acc*100:.2f}%")
    print(f"Gain FL vs single-site          : {(best_fl_acc - single_acc)*100:.2f}%")
    print(f"\nConvergence per round:")
    for i, acc in enumerate(round_results, 1):
        print(f"  Round {i}: {acc*100:.2f}%")