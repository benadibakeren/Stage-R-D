import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── 1. Charger les données texte ─────────────────────────
df = pd.read_csv("data/patient_text.csv")
print(f"Dataset: {len(df)} patients")
print(f"ADHD: {df['label'].sum()}, Control: {(df['label']==0).sum()}")

# ── 2. Split train/test ───────────────────────────────────
texts  = df["text"].tolist()
labels = df["label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# ── 3. Tokenizer ──────────────────────────────────────────
print("\nLoading BiomedBERT...")
MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

# ── 4. Dataset PyTorch ───────────────────────────────────
class EEGTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts  = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

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
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = EEGTextDataset(X_train, y_train, tokenizer)
test_dataset  = EEGTextDataset(X_test,  y_test,  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=8)

# ── 5. Training ───────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

print("\nTraining BiomedBERT...")
for epoch in range(10):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/10 — Loss: {avg_loss:.4f}")

# ── 6. Evaluation ─────────────────────────────────────────
print("\nEvaluating...")
model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=["Control", "ADHD"]
))

# ── 7. Sauvegarder le modèle ──────────────────────────────
model.save_pretrained("models/biomedbert_adhd")
tokenizer.save_pretrained("models/biomedbert_adhd")
print("\nModel saved to models/biomedbert_adhd/")