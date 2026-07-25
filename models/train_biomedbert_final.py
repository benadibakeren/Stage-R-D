import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# ── Reproducibility ───────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Data ──────────────────────────────────────────────────
df = pd.read_csv('data/patient_text.csv')
df["patient_id"] = df["ID"].str.rsplit("_", n=2).str[0]

print(f"Dataset: {len(df)} segments")
print(f"Patients: {df['patient_id'].nunique()}")
print(f"ADHD: {df['label'].sum()}, Control: {(df['label']==0).sum()}")

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Dataset ───────────────────────────────────────────────
class EEGTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(
            list(texts), max_length=max_len,
            padding="max_length", truncation=True,
            return_tensors="pt"
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids"     : self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label"         : torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── Patient-level evaluation via majority voting ──────────
def evaluate_patient_level(model, val_df):
    model.eval()
    loader = DataLoader(
        EEGTextDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer),
        batch_size=16
    )
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
    for pid, pred, label in zip(val_df["patient_id"], all_preds, val_df["label"]):
        if pid not in patient_preds:
            patient_preds[pid]  = []
            patient_labels[pid] = label
        patient_preds[pid].append(pred)

    final_preds  = [Counter(v).most_common(1)[0][0] for v in patient_preds.values()]
    final_labels = list(patient_labels.values())
    return accuracy_score(final_labels, final_preds), final_preds, final_labels

# ── GroupKFold — no data leakage across patients ──────────
texts      = df["text"].tolist()
labels     = df["label"].tolist()
groups     = df["patient_id"].tolist()

gkf        = GroupKFold(n_splits=5)
accuracies = []
best_acc   = 0.0
best_model_state = None

for fold, (train_idx, val_idx) in enumerate(gkf.split(texts, labels, groups)):
    print(f"\n{'='*40}")
    print(f"Fold {fold+1}/5")
    print(f"{'='*40}")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    print(f"Train: {train_df['patient_id'].nunique()} patients, {len(train_df)} segments")
    print(f"Val  : {val_df['patient_id'].nunique()} patients, {len(val_df)} segments")

    train_loader = DataLoader(
        EEGTextDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer),
        batch_size=16, shuffle=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    optimizer    = AdamW(model.parameters(), lr=3e-5)
    total_steps  = len(train_loader) * 10
    warmup_steps = int(0.1 * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids      =batch["input_ids"].to(device),
                attention_mask =batch["attention_mask"].to(device),
                labels         =batch["label"].to(device)
            )
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += outputs.loss.item()
        print(f"  Epoch {epoch+1}/10 — Loss: {total_loss/len(train_loader):.4f}")

    acc, final_preds, final_labels = evaluate_patient_level(model, val_df)
    accuracies.append(acc)
    print(f"\nFold {fold+1} patient-level accuracy: {acc*100:.2f}%")
    print(classification_report(final_labels, final_preds,
          target_names=["Control", "ADHD"], labels=[0, 1]))

    if acc > best_acc:
        best_acc = acc
        best_model_state = model.state_dict().copy()
        torch.save(best_model_state, "best_biomedbert.pt")
        print(f" Best model saved (fold {fold+1}: {acc*100:.2f}%)")

print(f"\n{'='*40}")
print(f"FINAL RESULTS — GroupKFold Patient-Level")
print(f"{'='*40}")
for i, acc in enumerate(accuracies):
    print(f"  Fold {i+1}: {acc*100:.2f}%")
print(f"\nMean accuracy : {np.mean(accuracies)*100:.2f}%")
print(f"Std           : {np.std(accuracies)*100:.2f}%")
print(f"Best model    : {best_acc*100:.2f}% → saved as best_biomedbert.pt")