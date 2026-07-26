import torch
import pandas as pd
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BioGptTokenizer,
    BioGptForCausalLM
)

# ── 1. Charger BiomedBERT avec les poids entraînés ───────
print("Chargement BiomedBERT...")
biomedbert_name      = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
biomedbert_tokenizer = AutoTokenizer.from_pretrained(biomedbert_name)
biomedbert_model     = AutoModelForSequenceClassification.from_pretrained(
    biomedbert_name, num_labels=2
)
# Charger les poids entraînés depuis best_biomedbert.pt
biomedbert_model.load_state_dict(
    torch.load("best_biomedbert.pt", map_location="cpu")
)
biomedbert_model.eval()
print("BiomedBERT chargé avec best_biomedbert.pt !")

# ── 2. Charger BioGPT ────────────────────────────────────
print("Chargement BioGPT...")
biogpt_tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
biogpt_model     = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
biogpt_model.eval()
print("BioGPT chargé !")

# ── 3. Charger les données ────────────────────────────────
df = pd.read_csv("data/patient_text.csv")
df["patient_id"] = df["ID"].str.rsplit("_", n=2).str[0]
print(f"Dataset: {df['patient_id'].nunique()} patients, {len(df)} segments")

# ── 4. Classification patient-level via vote majoritaire ──
def classify_patient(patient_id, df):
    patient_df = df[df["patient_id"] == patient_id]
    segments   = patient_df["text"].tolist()

    inputs = biomedbert_tokenizer(
        segments, max_length=256, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = biomedbert_model(**inputs)
        preds   = torch.argmax(outputs.logits, dim=1).numpy()

    vote = Counter(preds).most_common(1)[0][0]
    return "ADHD" if vote == 1 else "Control"

# ── 5. Génération rapport BioGPT ──────────────────────────
def generate_report(patient_id, tbr, faa, classification):
    tbr_status = "ELEVATED — indicates reduced cortical activation" if tbr > 2.5 else "NORMAL"
    faa_status = "ABNORMAL hemispheric asymmetry detected" if abs(faa) > 100 else "NORMAL"

    report_header = f"""CLINICAL EEG REPORT
Patient ID: {patient_id}
Classification: {classification}

NEUROPHYSIOLOGICAL FINDINGS:
- Frontal Theta/Beta Ratio (TBR): {tbr:.3f} — {tbr_status}
- Frontal Alpha Asymmetry (FAA): {faa:.3f} — {faa_status}

CLINICAL INTERPRETATION:
The neurophysiological profile indicates"""

    inputs = biogpt_tokenizer(report_header, return_tensors="pt")

    with torch.no_grad():
        outputs = biogpt_model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0
        )

    # Décoder uniquement les tokens générés — pas de duplication
    generated_ids  = outputs[0][inputs["input_ids"].shape[1]:]
    interpretation = biogpt_tokenizer.decode(generated_ids, skip_special_tokens=True)

    return report_header + interpretation

# ── 6. Pipeline complet ───────────────────────────────────
def full_pipeline(patient_id, df):
    print(f"\nTraitement patient {patient_id}...")

    # Classification via vote majoritaire
    classification = classify_patient(patient_id, df)
    print(f"BiomedBERT → {classification}")

    # Extraire TBR et FAA depuis le texte
    sample_text = df[df["patient_id"] == patient_id]["text"].iloc[0]
    tbr, faa = 0.0, 0.0
    for line in sample_text.split("\n"):
        if "TBR" in line:
            try: tbr = float(line.split(":")[-1].strip().split()[0])
            except: pass
        if "FAA" in line:
            try: faa = float(line.split(":")[-1].strip().split()[0])
            except: pass

    # Générer le rapport
    report = generate_report(patient_id, tbr, faa, classification)
    print(report)
    return report

# ── 7. Test sur deux patients ─────────────────────────────
# Un patient ADHD et un patient Control
adhd_patient    = df[df["label"]==1]["patient_id"].iloc[0]
control_patient = df[df["label"]==0]["patient_id"].iloc[0]

full_pipeline(adhd_patient, df)
full_pipeline(control_patient, df)