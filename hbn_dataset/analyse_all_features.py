import subprocess
import pandas as pd
import os

# IDs des 11 releases HBN-EEG
datasets = [f"ds00{i}" for i in range(5505, 5516)]

dfs = []

for ds in datasets:
    output_file = f"participants_{ds}.tsv"
    print(f"Téléchargement {ds}...")
    
    result = subprocess.run([
        "aws", "s3", "cp", "--no-sign-request",
        f"s3://openneuro.org/{ds}/participants.tsv",
        output_file
    ], capture_output=True, text=True)
    
    if os.path.exists(output_file):
        df = pd.read_csv(output_file, sep='\t')
        df['dataset'] = ds
        dfs.append(df)
        print(f"  → {len(df)} participants")
    else:
        print(f"  → échec")

# Concatener tout
df_all = pd.concat(dfs, ignore_index=True)
print(f"\nTotal participants toutes releases : {len(df_all)}")

# Filtrer utilisables
df_ok = df_all[
    (df_all['full_pheno'] == 'Yes') & 
    (df_all['RestingState'] == 'available')
]
print(f"Participants utilisables (full_pheno + RestingState) : {len(df_ok)}")
print(f"Potentiellement ADHD (attention > 0.5) : {(df_ok['attention'] > 0.5).sum()}")
print(f"Potentiellement contrôle (attention < -0.5) : {(df_ok['attention'] < -0.5).sum()}")