import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ── 1. Charger les features ──────────────────────────────
df_tbr = pd.read_csv("data/patient_tbr.csv")
df_features = pd.read_csv("data/patient_features.csv")

# Combiner TBR + features
df_all = pd.merge(df_tbr, df_features, on=["ID", "Class"])
print("Dataset shape:", df_all.shape)
print(df_all.head())

# ── 2. Préparer X et y ───────────────────────────────────
feature_cols = [col for col in df_all.columns 
                if col not in ["ID", "Class"]]
X = df_all[feature_cols].values
y = (df_all["Class"] == "ADHD").astype(int).values

print(f"\nFeatures: {feature_cols}")
print(f"X shape: {X.shape}")
print(f"y distribution: {np.bincount(y)}")

# ── 3. Split train/test ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ── 4. Normalisation ─────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ── 5. Dataset PyTorch ───────────────────────────────────
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = EEGDataset(X_train, y_train)
test_dataset = EEGDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ── 6. Modèle ────────────────────────────────────────────
class ADHDClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

model = ADHDClassifier(input_size=X_train.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"\nModel architecture:")
print(model)

# ── 7. Training loop ─────────────────────────────────────
print("\nTraining...")
for epoch in range(100):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(X_batch).squeeze()
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/100 — Loss: {total_loss/len(train_loader):.4f}")

# ── 8. Evaluation ────────────────────────────────────────
print("\nEvaluating...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        pred = model(X_batch).squeeze()
        pred_labels = (torch.sigmoid(pred) > 0.5).float()
        all_preds.extend(pred_labels.numpy())
        all_labels.extend(y_batch.numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, 
                           target_names=["Control", "ADHD"]))




# Random Forest avec cross-validation
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation sur 5 folds — plus fiable avec peu de données
scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
print(f"Cross-validation accuracy: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

# Entraîner sur tout le dataset
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["Control", "ADHD"]))

# Feature importance
importances = pd.Series(rf.feature_importances_, index=feature_cols)
print("\nTop features:")
print(importances.sort_values(ascending=False).head(10))