import pandas as pd


df = pd.read_csv('nasrabadi_raw/adhdata.csv')
print('Shape:', df.shape)
print('Patients:', df['ID'].nunique())
print('Lignes par patient:')
print(df.groupby('ID').size().describe())
