import mne
import os

# Lire le fichier EEG du premier patient
patient = 'sub-NDARAC688ZM5'
eeg_file = f'eeg_data/{patient}/{patient}_task-RestingState_eeg.set'

print(f"Lecture de {patient}...")
raw = mne.io.read_raw_eeglab(eeg_file, preload=True)

print(f"\nInfos générales :")
print(f"  Canaux : {len(raw.ch_names)}")
print(f"  Fréquence échantillonnage : {raw.info['sfreq']} Hz")
print(f"  Durée : {raw.times[-1]:.1f} secondes")
print(f"  Premiers canaux : {raw.ch_names[:5]}")