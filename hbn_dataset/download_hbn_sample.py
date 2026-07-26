import subprocess
import pandas as pd
import os

df = pd.read_csv('hbn_tsv/participants_ds005515.tsv', sep='\t')

df_ok = df[
    (df['full_pheno'] == 'Yes') & 
    (df['RestingState'] == 'available')
]

# Prendre 5 ADHD et 5 contrôles
adhd = df_ok[df_ok['attention'] > 0.5].head(5)
ctrl = df_ok[df_ok['attention'] < -0.5].head(5)
sample = pd.concat([adhd, ctrl])

print(f"ADHD : {len(adhd)}, Contrôles : {len(ctrl)}")
print(sample[['participant_id', 'attention']])

os.makedirs('eeg_data', exist_ok=True)

for _, row in sample.iterrows():
    pid = row['participant_id']
    print(f"\nTéléchargement {pid}...")
    subprocess.run([
        "aws", "s3", "sync", "--no-sign-request",
        f"s3://openneuro.org/ds005515/{pid}/eeg/",
        f"eeg_data/{pid}/",
        "--exclude", "*",
        "--include", "*task-RestingState*"
    ])

print("\nTerminé !")