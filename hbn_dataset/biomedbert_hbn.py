import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# Charger les données
df = pd.read_csv('hbn_text.csv')
print(f"Dataset : {len(df)} patients, {df['label'].sum()} ADHD, {(df['label']==0).sum()} contrôles")

# Dataset PyTorch
class EEGDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, 
            max_length=max_len, return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

# Modèle et tokenizer
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)

texts = df['text'].tolist()
labels = df['label'].tolist()

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
    print(f"\nFold {fold+1}/5...")

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]

    train_dataset = EEGDataset(train_texts, train_labels, tokenizer)
    val_dataset = EEGDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Entraînement
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()

    # Évaluation
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(**batch)
            preds.extend(outputs.logits.argmax(-1).tolist())
            true.extend(batch['labels'].tolist())

    acc = accuracy_score(true, preds)
    accuracies.append(acc)
    print(f"  Accuracy fold {fold+1} : {acc:.4f}")

print(f"\nMean accuracy : {np.mean(accuracies):.4f}")
print(f"Std : {np.std(accuracies):.4f}")