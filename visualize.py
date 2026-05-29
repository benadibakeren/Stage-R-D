import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Charger les features
df_features = pd.read_csv("data/patient_features.csv")

adhd = df_features[df_features["Class"] == "ADHD"]
control = df_features[df_features["Class"] == "Control"]

# ── 1. Boxplot Fz Standard Deviation ─────────────────────
plt.figure(figsize=(8, 5))
plt.boxplot([adhd["Fz_std"].values, control["Fz_std"].values],
            labels=["ADHD", "Control"],
            patch_artist=True,
            boxprops=dict(facecolor="lightcoral", color="red"),
            medianprops=dict(color="darkred", linewidth=2))
plt.title("Fz Standard Deviation — ADHD vs Control")
plt.ylabel("Std (µV)")
plt.grid(axis="y", alpha=0.3)
plt.savefig("boxplot_fz_std.png")
print("Saved: boxplot_fz_std.png")

# ── 2. Boxplot Fz Mean ───────────────────────────────────
plt.figure(figsize=(8, 5))
plt.boxplot([adhd["Fz_mean"].values, control["Fz_mean"].values],
            labels=["ADHD", "Control"],
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="blue"),
            medianprops=dict(color="darkblue", linewidth=2))
plt.title("Fz Mean Amplitude — ADHD vs Control")
plt.ylabel("Mean (µV)")
plt.grid(axis="y", alpha=0.3)
plt.savefig("boxplot_fz_mean.png")
print("Saved: boxplot_fz_mean.png")

# ── 3. Comparaison de toutes les features ────────────────
features = ["Fz_mean", "Fz_std", "F3_mean", "F3_std", "Fp1_mean", "Fp1_std"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, feature in enumerate(features):
    axes[i].boxplot(
        [adhd[feature].values, control[feature].values],
        labels=["ADHD", "Control"],
        patch_artist=True,
        boxprops=dict(facecolor="lightyellow"),
        medianprops=dict(color="orange", linewidth=2)
    )
    axes[i].set_title(feature)
    axes[i].grid(axis="y", alpha=0.3)

plt.suptitle("EEG Features — ADHD vs Control", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("all_features_comparison.png")
print("Saved: all_features_comparison.png")

# ── 4. Statistiques simples ──────────────────────────────
print("\n--- ADHD vs Control Feature Comparison ---")
for col in ["Fz_mean", "Fz_std", "F3_mean", "F3_std"]:
    print(f"\n{col}:")
    print(f"  ADHD    → mean={adhd[col].mean():.2f}, std={adhd[col].std():.2f}")
    print(f"  Control → mean={control[col].mean():.2f}, std={control[col].std():.2f}")