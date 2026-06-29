import pandas as pd
import numpy as np
from scipy.signal import welch

def extract_features(df, sfreq=128):
    """
    Extrait TBR, FAA et puissances spectrales
    depuis le signal EEG brut Nasrabadi
    """
    frontal = ['Fz', 'F3', 'F4', 'Fp1', 'Fp2']
    records = []

    for patient_id, group in df.groupby('ID'):
        label = group['Class'].iloc[0]
        signal = group[frontal].values

        theta_p, beta_p, alpha_p = [], [], []

        for ch in range(signal.shape[1]):
            freqs, psd = welch(signal[:, ch], fs=sfreq, nperseg=256)
            theta_p.append(np.trapezoid(
                psd[(freqs>=4)&(freqs<=8)], freqs[(freqs>=4)&(freqs<=8)]))
            beta_p.append(np.trapezoid(
                psd[(freqs>=13)&(freqs<=30)], freqs[(freqs>=13)&(freqs<=30)]))
            alpha_p.append(np.trapezoid(
                psd[(freqs>=8)&(freqs<=13)], freqs[(freqs>=8)&(freqs<=13)]))

        tbr = np.mean(theta_p) / np.mean(beta_p)
        faa = alpha_p[frontal.index('F4')] - alpha_p[frontal.index('F3')]

        records.append({
            'ID'        : patient_id,
            'label'     : 1 if label == 'ADHD' else 0,
            'TBR'       : tbr,
            'FAA'       : faa,
            'theta_mean': np.mean(theta_p),
            'beta_mean' : np.mean(beta_p),
            'alpha_mean': np.mean(alpha_p)
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    df_raw = pd.read_csv('nasrabadi_raw/adhdata.csv')
    df_features = extract_features(df_raw)
    df_features.to_csv('data/patient_features.csv', index=False)
    print(f"Features extraites : {len(df_features)} patients")
    print(df_features.head())