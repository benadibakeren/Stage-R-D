import mne
import numpy as np
import pandas as pd
import os

def extract_features(raw):
    # Filtrage basique
    raw.filter(1., 40., fir_design='firwin')
    
    # Calcul de la puissance spectrale par bande
    psds, freqs = mne.time_frequency.psd_array_welch(
        raw.get_data(), 
        sfreq=raw.info['sfreq'],
        fmin=1., fmax=40.,
        n_fft=2048
    )
    
    # Définir les bandes
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30)
    }
    
    features = {}
    for band, (fmin, fmax) in bands.items():
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        features[f'{band}_power'] = psds[:, idx].mean()
    
    # TBR
    features['TBR'] = features['theta_power'] / features['beta_power']
    
    return features

# Traiter les 5 patients
results = []
df_labels = pd.read_csv('hbn_tsv/participants_ds005515.tsv', sep='\t')

for patient_folder in os.listdir('eeg_data'):
    eeg_file = f'eeg_data/{patient_folder}/{patient_folder}_task-RestingState_eeg.set'
    
    if not os.path.exists(eeg_file):
        continue
    
    print(f"Traitement {patient_folder}...")
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
    features = extract_features(raw)
    features['participant_id'] = patient_folder
    
    # Ajouter le label attention
    label = df_labels[df_labels['participant_id'] == patient_folder]['attention'].values
    features['attention'] = label[0] if len(label) > 0 else None
    features['label'] = 1 if features['attention'] > 0.5 else 0
    
    results.append(features)

df = pd.DataFrame(results)
df.to_csv('hbn_features.csv', index=False)
print("\nFeatures extraites :")
print(df)