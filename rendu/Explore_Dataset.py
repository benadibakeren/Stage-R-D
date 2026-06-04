import pandas as pd
import matplotlib.pyplot as plt

# Lire le fichier CSV
df = pd.read_csv("data/adhdata.csv")

# ── 1. Vue générale ──────────────────────────────────────
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# ── 2. Distribution des classes ──────────────────────────
print("\nClass distribution:")
print(df["Class"].value_counts())

# ── 3. Vérifier les valeurs manquantes ───────────────────
print("\nMissing values:")
print(df.isnull().sum())

# ── 4. Statistiques de base ──────────────────────────────
print("\nBasic statistics:")
print(df.describe())

# ── 5. Séparer ADHD et Control ───────────────────────────
adhd = df[df["Class"] == "ADHD"]
control = df[df["Class"] == "Control"]
print(f"\nADHD samples: {len(adhd)}")
print(f"Control samples: {len(control)}")

# ── 6. Visualiser un signal EEG ──────────────────────────
plt.figure(figsize=(12, 4))
plt.plot(adhd["Fz"].values[:500], label="ADHD - Fz", color="red", alpha=0.7)
plt.plot(control["Fz"].values[:500], label="Control - Fz", color="blue", alpha=0.7)
plt.title("EEG Signal - Fz Channel (first 500 samples)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("eeg_signal.png")
print("Plot saved as eeg_signal.png")

print("Nombre de patients:", df["ID"].nunique())
print("Patients par classe:")
print(df.groupby("Class")["ID"].nunique())