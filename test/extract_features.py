import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def extract_features(patient_data):
    """Extrait les features EEG pour un patient"""
    features = {}
    
    # Canaux frontaux importants pour l'ADHD
    frontal_channels = ["Fz", "F3", "F4", "Fp1", "Fp2"]
    
    for channel in frontal_channels:
        signal = patient_data[channel].values
        
        # Statistiques simples
        features[f"{channel}_mean"] = np.mean(signal)
        features[f"{channel}_std"] = np.std(signal)
        features[f"{channel}_max"] = np.max(signal)
        
    return features

# Extraire les features pour chaque patient
df = pd.read_csv("data/adhdata.csv")

patient_features = []

for patient_id in df["ID"].unique():
    patient_data = df[df["ID"] == patient_id]
    label = patient_data["Class"].iloc[0]
    
    features = extract_features(patient_data)
    features["ID"] = patient_id
    features["Class"] = label
    patient_features.append(features)

# Créer un dataframe de features
df_features = pd.DataFrame(patient_features)
print("Features shape:", df_features.shape)
print(df_features.head())

df_features.to_csv("data/patient_features.csv", index=False)
print("Features saved to data/patient_features.csv")


adhd = df_features[df_features["Class"] == "ADHD"]
control = df_features[df_features["Class"] == "Control"]

plt.figure(figsize=(10, 4))
plt.boxplot([adhd["Fz_std"].values, control["Fz_std"].values],
            labels=["ADHD", "Control"])
plt.title("Fz Standard Deviation — ADHD vs Control")
plt.ylabel("Std (µV)")
plt.savefig("adhd_vs_control.png")
print("Plot saved!")