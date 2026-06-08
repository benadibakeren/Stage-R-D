import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('data/patient_text.csv')
print(f"Dataset: {len(df)} patients")
print(f"ADHD: {df['label'].sum()}, Control: {(df['label']==0).sum()}")

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class EEGTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = list(texts)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

texts = df["text"].tolist()
labels = df["label"].tolist()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
    print(f"\nFold {fold+1}/5...")

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]

    train_dataset = EEGTextDataset(train_texts, train_labels, tokenizer)
    val_dataset = EEGTextDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch_labels
            )
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()

        print(f"  Epoch {epoch+1}/10 — Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            true.extend(batch_labels.cpu().numpy())

    acc = accuracy_score(true, preds)
    accuracies.append(acc)
    print(f"  Accuracy fold {fold+1} : {acc:.4f}")
    print(classification_report(true, preds,
          target_names=['Control', 'ADHD'], labels=[0, 1]))

print(f"\nMean accuracy : {np.mean(accuracies):.4f}")
print(f"Std : {np.std(accuracies):.4f}")