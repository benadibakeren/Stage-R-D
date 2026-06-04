import subprocess
import pandas as pd
import mne
import numpy as np
import os
import shutil

def extract_features(eeg_file):
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
    raw.filter(1., 40., fir_design='firwin', verbose=False)
    
    psds, freqs = mne.time_frequency.psd_array_welch(
        raw.get_data(),
        sfreq=raw.info['sfreq'],
        fmin=1., fmax=40.,
        n_fft=2048,
        verbose=False
    )
    
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
    
    features['TBR'] = features['theta_power'] / features['beta_power']
    return features

# Charger tous les participants disponibles
all_dfs = []
for i in range(5505, 5516):
    tsv = f'hbn_tsv/participants_ds00{i}.tsv'
    if os.path.exists(tsv):
        df = pd.read_csv(tsv, sep='\t')
        df['dataset'] = f'ds00{i}'
        all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)

# Filtrer utilisables
df_ok = df_all[
    (df_all['full_pheno'] == 'Yes') &
    (df_all['RestingState'] == 'available')
].copy()

# Équilibrer ADHD vs contrôle
adhd = df_ok[df_ok['attention'] > 0.5].head(100)
ctrl = df_ok[df_ok['attention'] < -0.5].head(100)
sample = pd.concat([adhd, ctrl]).reset_index(drop=True)

print(f"Patients à traiter : {len(sample)} ({len(adhd)} ADHD, {len(ctrl)} contrôles)")

# Traitement par batch de 20
results = []
batch_size = 20

for i in range(0, len(sample), batch_size):
    batch = sample.iloc[i:i+batch_size]
    print(f"\nBatch {i//batch_size + 1} ({i+1} à {min(i+batch_size, len(sample))})...")

    # Télécharger
    for _, row in batch.iterrows():
        pid = row['participant_id']
        ds = row['dataset']
        subprocess.run([
            "aws", "s3", "sync", "--no-sign-request",
            f"s3://openneuro.org/{ds}/{pid}/eeg/",
            f"eeg_data/{pid}/",
            "--exclude", "*",
            "--include", "*RestingState*"
        ], capture_output=True)

    # Extraire features
    for _, row in batch.iterrows():
        pid = row['participant_id']
        eeg_file = f"eeg_data/{pid}/{pid}_task-RestingState_eeg.set"
        
        if not os.path.exists(eeg_file):
            print(f"  {pid} — fichier manquant, ignoré")
            continue
        
        try:
            features = extract_features(eeg_file)
            features['participant_id'] = pid
            features['attention'] = row['attention']
            features['label'] = 1 if row['attention'] > 0.5 else 0
            results.append(features)
            print(f"  {pid} ✓")
        except Exception as e:
            print(f"  {pid} — erreur : {e}")

    # Supprimer les fichiers bruts pour libérer l'espace
    shutil.rmtree('eeg_data', ignore_errors=True)
    os.makedirs('eeg_data', exist_ok=True)

    # Sauvegarder au fur et à mesure
    pd.DataFrame(results).to_csv('hbn_features_200.csv', index=False)
    print(f"  → {len(results)} patients traités au total")

print(f"\nTerminé ! {len(results)} patients dans hbn_features_200.csv")