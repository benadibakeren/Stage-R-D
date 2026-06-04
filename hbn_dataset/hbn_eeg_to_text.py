import pandas as pd

df = pd.read_csv('hbn_features.csv')

def features_to_text(row):
    return (
        f"EEG resting state analysis: "
        f"delta power {row['delta_power']:.2e}, "
        f"theta power {row['theta_power']:.2e}, "
        f"alpha power {row['alpha_power']:.2e}, "
        f"beta power {row['beta_power']:.2e}, "
        f"theta-beta ratio {row['TBR']:.3f}. "
        f"Patient age range: child/adolescent. "
        f"Recording: 128 channels, 500Hz, resting state eyes open."
    )

df['text'] = df.apply(features_to_text, axis=1)

# Sauvegarder
df[['participant_id', 'text', 'label']].to_csv('hbn_text.csv', index=False)

print("Exemples générés :")
for _, row in df.iterrows():
    print(f"\n[Label {row['label']}] {row['text'][:100]}...")