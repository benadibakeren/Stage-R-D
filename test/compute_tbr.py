import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

# Charger les données brutes
df = pd.read_csv("data/adhdata.csv")

# Fréquence d'échantillonnage (128 Hz selon le dataset)
FS = 128

def compute_band_power(signal_data, fs, low_freq, high_freq):
    """Calcule la puissance d'une bande de fréquence"""
    # FFT
    freqs, psd = scipy_signal.welch(signal_data, fs=fs, nperseg=fs*2)
    
    # Sélectionner les fréquences de la bande
    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
    
    # Puissance = intégrale de la PSD sur la bande
    band_power = np.trapezoid(psd[band_mask], freqs[band_mask])
    
    return band_power

def compute_tbr_for_patient(patient_data, fs=128):
    """Calcule le Theta/Beta Ratio pour un patient"""
    
    # Canaux frontaux
    frontal_channels = ["Fz", "F3", "F4"]
    
    theta_powers = []
    beta_powers = []
    
    for channel in frontal_channels:
        sig = patient_data[channel].values.astype(float)
        
        # Bandes de fréquence
        theta = compute_band_power(sig, fs, 4, 8)   # Theta: 4-8 Hz
        beta = compute_band_power(sig, fs, 13, 30)   # Beta: 13-30 Hz
        
        theta_powers.append(theta)
        beta_powers.append(beta)
    
    # Moyenne sur les canaux frontaux
    mean_theta = np.mean(theta_powers)
    mean_beta = np.mean(beta_powers)
    
    # TBR
    tbr = mean_theta / mean_beta if mean_beta > 0 else 0
    
    return {
        "theta_power": mean_theta,
        "beta_power": mean_beta,
        "tbr": tbr
    }

# Calculer le TBR pour chaque patient
print("Computing TBR for each patient...")
results = []

for patient_id in df["ID"].unique():
    patient_data = df[df["ID"] == patient_id]
    label = patient_data["Class"].iloc[0]
    
    tbr_features = compute_tbr_for_patient(patient_data)
    tbr_features["ID"] = patient_id
    tbr_features["Class"] = label
    results.append(tbr_features)
    print(f"  {patient_id} ({label}) → TBR = {tbr_features['tbr']:.3f}")

# Créer le dataframe
df_tbr = pd.DataFrame(results)
print("\nTBR Dataset:")
print(df_tbr.head(10))

# Sauvegarder
df_tbr.to_csv("data/patient_tbr.csv", index=False)
print("\nSaved: data/patient_tbr.csv")

# ── Visualiser le TBR ────────────────────────────────────
adhd = df_tbr[df_tbr["Class"] == "ADHD"]
control = df_tbr[df_tbr["Class"] == "Control"]

print(f"\nADHD   → TBR mean = {adhd['tbr'].mean():.3f}")
print(f"Control → TBR mean = {control['tbr'].mean():.3f}")

plt.figure(figsize=(8, 5))
plt.boxplot(
    [adhd["tbr"].values, control["tbr"].values],
    labels=["ADHD", "Control"],
    patch_artist=True,
    boxprops=dict(facecolor="lightcoral", color="red"),
    medianprops=dict(color="darkred", linewidth=2)
)
plt.title("Theta/Beta Ratio (TBR) — ADHD vs Control")
plt.ylabel("TBR")
plt.grid(axis="y", alpha=0.3)
plt.savefig("tbr_comparison.png")
print("Saved: tbr_comparison.png")