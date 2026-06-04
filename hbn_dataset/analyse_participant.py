import pandas as pd
import numpy as np

df = pd.read_csv('participants.tsv', sep='\t')

print("Colonnes disponibles :")
print(df.columns.tolist())

print(f"\nNombre total de participants : {len(df)}")

print("\nAperçu des données :")
print(df.head(10))

print("\nStatistiques de la colonne attention :")
print(df['attention'].describe())

print("\nDistribution de full_pheno :")
print(df['full_pheno'].value_counts())

print("\nDisponibilité RestingState :")
print(df['RestingState'].value_counts())

# Participants avec phénotypage complet ET RestingState disponible
df_complet = df[(df['full_pheno'] == 'Yes') & (df['RestingState'] == 'available')]
print(f"\nParticipants avec full_pheno + RestingState disponible : {len(df_complet)}")

print("\nStatistiques attention pour ce sous-groupe :")
print(df_complet['attention'].describe())

print(f"\nScore attention > 0.5 (potentiellement ADHD) : {(df_complet['attention'] > 0.5).sum()}")
print(f"Score attention < -0.5 (potentiellement contrôle) : {(df_complet['attention'] < -0.5).sum()}")