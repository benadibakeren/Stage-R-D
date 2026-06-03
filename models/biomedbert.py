import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score

# ── 1. Charger les données ────────────────────────────────
df = pd.read_csv("patient_text.csv")
texts  = df["text"].tolist()
labels = df["label"].tolist()

# ── 2. Dataset PyTorch ────────────────────────────────────
MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

class EEGTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── 3. Cross-validation ───────────────────────────────────
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
texts_np = np.array(texts)
labels_np = np.array(labels)

fold_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(texts_np, labels_np)):
    print(f"\n--- Fold {fold+1}/5 ---")

    X_train = texts_np[train_idx].tolist()
    X_test  = texts_np[test_idx].tolist()
    y_train = labels_np[train_idx].tolist()
    y_test  = labels_np[test_idx].tolist()

    train_loader = DataLoader(EEGTextDataset(X_train, y_train, tokenizer), batch_size=8, shuffle=True)
    test_loader  = DataLoader(EEGTextDataset(X_test,  y_test,  tokenizer), batch_size=8)

    # Nouveau modèle à chaque fold
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Entraînement
    for epoch in range(20):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                labels         = batch["label"].to(device)
            )
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()
        print(f"  Epoch {epoch+1}/20 — Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device)
            )
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    acc = accuracy_score(all_labels, all_preds)
    fold_scores.append(acc)
    print(f"  Fold {fold+1} Accuracy: {acc*100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=["Control", "ADHD"]))

# ── 4. Résultat final ─────────────────────────────────────
print(f"\n{'='*40}")
print(f"Mean Accuracy: {np.mean(fold_scores)*100:.2f}%")
print(f"Std:           {np.std(fold_scores)*100:.2f}%")
print(f"Per fold:      {[f'{s*100:.1f}%' for s in fold_scores]}")