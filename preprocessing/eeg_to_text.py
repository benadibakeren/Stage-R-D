import pandas as pd
import numpy as np
from scipy.signal import welch

def extract_windows(df, window_sec=30, sfreq=128):
    """
    Découpe le signal EEG en fenêtres temporelles
    et sérialise chaque fenêtre en texte structuré
    """
    window_samples = window_sec * sfreq
    frontal = ['Fz', 'F3', 'F4', 'Fp1', 'Fp2']
    records = []

    for patient_id, group in df.groupby('ID'):
        label = group['Class'].iloc[0]
        signal = group[frontal].values
        n_windows = len(signal) // window_samples

        for w in range(n_windows):
            start = w * window_samples
            end   = start + window_samples
            segment = signal[start:end]

            theta_p, beta_p, alpha_p = [], [], []

            for ch in range(segment.shape[1]):
                freqs, psd = welch(
                    segment[:, ch], fs=sfreq,
                    nperseg=min(256, window_samples)
                )
                theta_p.append(np.trapezoid(
                    psd[(freqs>=4)&(freqs<=8)],
                    freqs[(freqs>=4)&(freqs<=8)]))
                beta_p.append(np.trapezoid(
                    psd[(freqs>=13)&(freqs<=30)],
                    freqs[(freqs>=13)&(freqs<=30)]))
                alpha_p.append(np.trapezoid(
                    psd[(freqs>=8)&(freqs<=13)],
                    freqs[(freqs>=8)&(freqs<=13)]))

            tbr = np.mean(theta_p) / np.mean(beta_p)
            faa = alpha_p[frontal.index('F4')] - alpha_p[frontal.index('F3')]

            text = f"""Patient ID: PATIENT_ANONYMIZED
Frontal Theta/Beta Ratio (TBR): {tbr:.3f}
Frontal Alpha Asymmetry (FAA): {faa:.3f}
Fz Theta Power: {theta_p[0]:.4f} uV2/Hz
Fz Beta Power: {beta_p[0]:.4f} uV2/Hz
F3 Theta Power: {theta_p[1]:.4f} uV2/Hz
F4 Theta Power: {theta_p[2]:.4f} uV2/Hz"""

            records.append({
                'ID'    : f"{patient_id}_seg{w}",
                'label' : 1 if label == 'ADHD' else 0,
                'Class' : label,
                'text'  : text
            })

    return pd.DataFrame(records)

if __name__ == "__main__":
    df_raw = pd.read_csv('nasrabadi_raw/adhdata.csv')

    for window_sec in [10, 20, 30, 60, 120]:
        df_window = extract_windows(df_raw, window_sec=window_sec)
        output = f'data/patient_text_w{window_sec}s.csv'
        df_window.to_csv(output, index=False)
        print(f"Fenêtre {window_sec}s : {len(df_window)} segments → {output}")


# Générer aussi patient_text.csv (fenêtre 30s, format standard)
df_30s = extract_windows(df_raw, window_sec=30)
# Renommer les segments au format attendu
df_30s["ID"] = df_30s["ID"].str.replace("_seg", "_segment_")
# Réordonner les colonnes
df_30s = df_30s[["ID", "Class", "label", "text"]]
df_30s.to_csv("data/patient_text.csv", index=False)
print(f"patient_text.csv généré : {len(df_30s)} segments")