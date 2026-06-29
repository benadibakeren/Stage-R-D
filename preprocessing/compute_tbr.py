import pandas as pd
import numpy as np
from scipy.signal import welch

def compute_tbr_patient(df, sfreq=128):
    """
    Calcule le TBR moyen par patient
    depuis le signal EEG brut
    """
    frontal = ['Fz', 'F3', 'F4', 'Fp1', 'Fp2']
    records = []

    for patient_id, group in df.groupby('ID'):
        label  = group['Class'].iloc[0]
        signal = group[frontal].values

        theta_p, beta_p = [], []

        for ch in range(signal.shape[1]):
            freqs, psd = welch(signal[:, ch], fs=sfreq, nperseg=256)
            theta_p.append(np.trapezoid(
                psd[(freqs>=4)&(freqs<=8)], freqs[(freqs>=4)&(freqs<=8)]))
            beta_p.append(np.trapezoid(
                psd[(freqs>=13)&(freqs<=30)], freqs[(freqs>=13)&(freqs<=30)]))

        tbr = np.mean(theta_p) / np.mean(beta_p)

        records.append({
            'ID'   : patient_id,
            'label': 1 if label == 'ADHD' else 0,
            'Class': label,
            'TBR'  : tbr
        })

    df_tbr = pd.DataFrame(records)
    print(f"\nTBR moyen ADHD    : {df_tbr[df_tbr['label']==1]['TBR'].mean():.3f}")
    print(f"TBR moyen Control : {df_tbr[df_tbr['label']==0]['TBR'].mean():.3f}")
    return df_tbr

if __name__ == "__main__":
    df_raw = pd.read_csv('nasrabadi_raw/adhdata.csv')
    df_tbr = compute_tbr_patient(df_raw)
    df_tbr.to_csv('data/patient_tbr.csv', index=False)
    print(f"TBR calculé pour {len(df_tbr)} patients")