import pandas as pd
import numpy as np
from scipy.signal import welch

def extract_windows(df, window_sec, sfreq=128):
    """Extrait les features EEG pour une taille de fenêtre donnée"""
    
    window_samples = window_sec * sfreq
    frontal = ['Fz', 'F3', 'F4', 'Fp1', 'Fp2']
    records = []
    
    for patient_id, group in df.groupby('ID'):
        label = group['Class'].iloc[0]
        signal = group[frontal].values
        n_samples = len(signal)
        n_windows = n_samples // window_samples
        
        for w in range(n_windows):
            start = w * window_samples
            end = start + window_samples
            segment = signal[start:end]
            
            theta_p, beta_p, alpha_p = [], [], []
            
            for ch in range(segment.shape[1]):
                freqs, psd = welch(segment[:, ch], fs=sfreq, nperseg=min(256, window_samples))
                theta_p.append(np.trapezoid(psd[(freqs>=4)&(freqs<=8)], freqs[(freqs>=4)&(freqs<=8)]))
                beta_p.append(np.trapezoid(psd[(freqs>=13)&(freqs<=30)], freqs[(freqs>=13)&(freqs<=30)]))
                alpha_p.append(np.trapezoid(psd[(freqs>=8)&(freqs<=13)], freqs[(freqs>=8)&(freqs<=13)]))
            
            tbr = np.mean(theta_p) / np.mean(beta_p)
            faa = alpha_p[frontal.index('F4')] - alpha_p[frontal.index('F3')]
            
            text = f"""Patient ID: {patient_id}_seg{w}
Frontal Theta/Beta Ratio (TBR): {tbr:.3f}
Frontal Alpha Asymmetry (FAA): {faa:.3f}
Fz Theta Power: {theta_p[0]:.4f} uV2/Hz
Fz Beta Power: {beta_p[0]:.4f} uV2/Hz
F3 Theta Power: {theta_p[1]:.4f} uV2/Hz
F4 Theta Power: {theta_p[2]:.4f} uV2/Hz"""
            
            records.append({
                'ID': f"{patient_id}_w{window_sec}s_seg{w}",
                'patient_id': patient_id,
                'label': 1 if label == 'ADHD' else 0,
                'text': text,
                'window_sec': window_sec
            })
    
    return pd.DataFrame(records)

# Charger les données brutes
df_raw = pd.read_csv('nasrabadi_raw/adhdata.csv')

# Générer les datasets pour chaque taille de fenêtre
WINDOW_SIZES = [10, 20, 30, 60, 120]

for window_sec in WINDOW_SIZES:
    print(f"\nFenêtre {window_sec} sec...")
    df_window = extract_windows(df_raw, window_sec)
    
    output_path = f"data/patient_text_w{window_sec}s.csv"
    df_window.to_csv(output_path, index=False)
    
    n_adhd = (df_window['label']==1).sum()
    n_ctrl = (df_window['label']==0).sum()
    print(f"  {len(df_window)} segments — ADHD: {n_adhd}, Control: {n_ctrl}")
    print(f"  Sauvegardé : {output_path}")

