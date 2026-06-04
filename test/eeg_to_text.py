import pandas as pd
import numpy as np
from scipy import signal as scipy_signal

# ── 1. Charger les données ────────────────────────────────
df = pd.read_csv("data/adhdata.csv")
FS = 128

def compute_band_power(signal_data, fs, low_freq, high_freq):
    freqs, psd = scipy_signal.welch(signal_data, fs=fs, nperseg=fs*2)
    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
    return np.trapezoid(psd[band_mask], freqs[band_mask])

def extract_eeg_features(patient_data, fs=128):
    """Extrait toutes les features EEG d'un patient"""
    frontal = ["Fz", "F3", "F4", "Fp1", "Fp2"]
    features = {}
    
    for ch in frontal:
        sig = patient_data[ch].values.astype(float)
        features[f"{ch}_theta"] = compute_band_power(sig, fs, 4, 8)
        features[f"{ch}_beta"]  = compute_band_power(sig, fs, 13, 30)
        features[f"{ch}_alpha"] = compute_band_power(sig, fs, 8, 13)
    
    # TBR moyen sur canaux frontaux
    mean_theta = np.mean([features[f"{ch}_theta"] for ch in frontal])
    mean_beta  = np.mean([features[f"{ch}_beta"]  for ch in frontal])
    features["tbr"] = mean_theta / mean_beta if mean_beta > 0 else 0
    
    # Asymétrie frontale alpha (F4 - F3)
    features["faa"] = features["F4_alpha"] - features["F3_alpha"]
    
    return features

def features_to_text(patient_id, features, label=None):
    """Convertit les features EEG en texte structuré pour le LLM"""
    
    text = f"""Patient ID: {patient_id}
Frontal Theta/Beta Ratio (TBR): {features['tbr']:.3f}
Frontal Alpha Asymmetry (FAA): {features['faa']:.3f}
Fz Theta Power: {features['Fz_theta']:.2f} uV2/Hz
Fz Beta Power: {features['Fz_beta']:.2f} uV2/Hz
Fz Alpha Power: {features['Fz_alpha']:.2f} uV2/Hz
F3 Theta Power: {features['F3_theta']:.2f} uV2/Hz
F4 Theta Power: {features['F4_theta']:.2f} uV2/Hz"""
    
    if label:
        text += f"\nDiagnosis: {label}"
    
    return text

# ── 2. Convertir tous les patients ───────────────────────
print("Extracting features and converting to text...")
records = []

for patient_id in df["ID"].unique():
    patient_data = df[df["ID"] == patient_id]
    label = patient_data["Class"].iloc[0]
    
    features = extract_eeg_features(patient_data)
    text = features_to_text(patient_id, features)  
    records.append({
        "ID": patient_id,
        "Class": label,
        "label": 1 if label == "ADHD" else 0,
        "text": text
    })
    print(f"  {patient_id} ({label}) → text generated")

# ── 3. Sauvegarder ───────────────────────────────────────
df_text = pd.DataFrame(records)
df_text.to_csv("data/patient_text.csv", index=False)
print(f"\nDone! {len(df_text)} patients converted.")
print("\nExample text:")
print(df_text["text"].iloc[0])