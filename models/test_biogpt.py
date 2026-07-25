"""
BioGPT Clinical Report Generation
===================================
Generates natural language clinical reports from EEG features
and BiomedBERT classification output.

Fixes applied:
- Reads data from patient_text.csv and best_biomedbert.pt
- Decodes only generated tokens (no duplication)
- Conditional "elevated" based on actual TBR value
- Fixed v10p/v11p comment inconsistency
- Pipeline properly chained: Step 2 output → Step 3 input

Author: Keren Benadiba
Institution: UMONS, Service SEMi
"""

import torch
import pandas as pd
from collections import Counter
from transformers import (
    BioGptTokenizer, BioGptForCausalLM,
    AutoTokenizer, AutoModelForSequenceClassification
)

# ── Load BiomedBERT ───────────────────────────────────────
BIOMEDBERT_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
BIOGPT_NAME     = "microsoft/biogpt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load BiomedBERT from saved checkpoint
bert_tokenizer = AutoTokenizer.from_pretrained(BIOMEDBERT_NAME)
bert_model     = AutoModelForSequenceClassification.from_pretrained(
    BIOMEDBERT_NAME, num_labels=2
).to(device)
bert_model.load_state_dict(
    torch.load("best_biomedbert.pt", map_location=device)
)
bert_model.eval()
print("✅ BiomedBERT loaded from best_biomedbert.pt")

# Load BioGPT
gpt_tokenizer = BioGptTokenizer.from_pretrained(BIOGPT_NAME)
gpt_model     = BioGptForCausalLM.from_pretrained(BIOGPT_NAME)
gpt_model.eval()
print("✅ BioGPT loaded")

# ── Classify patient via majority voting ──────────────────
def classify_patient(patient_id, df):
    """
    Classifies a patient using BiomedBERT with majority voting
    across all their segments.
    Returns: (classification_str, tbr, faa)
    """
    patient_df = df[df["patient_id"] == patient_id]
    if len(patient_df) == 0:
        raise ValueError(f"Patient {patient_id} not found in dataset")

    segments = patient_df["text"].tolist()
    inputs   = bert_tokenizer(
        segments, max_length=256, padding="max_length",
        truncation=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)
        preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()

    # Majority vote
    vote           = Counter(preds).most_common(1)[0][0]
    classification = "ADHD" if vote == 1 else "Control"

    # Extract TBR and FAA from text
    sample_text = patient_df["text"].iloc[0]
    tbr, faa = 0.0, 0.0
    for line in sample_text.split("\n"):
        if "TBR" in line:
            try: tbr = float(line.split(":")[-1].strip().split()[0])
            except: pass
        if "FAA" in line:
            try: faa = float(line.split(":")[-1].strip().split()[0])
            except: pass

    return classification, tbr, faa

# ── Generate clinical report ──────────────────────────────
def generate_clinical_report(patient_id, tbr, faa, classification):
    """
    Generates a structured clinical report using BioGPT.
    Only generated tokens are decoded — no duplication.
    TBR/FAA interpretation is conditional on actual values.
    """
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

    inputs = gpt_tokenizer(report_header, return_tensors="pt")

    with torch.no_grad():
        outputs = gpt_model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0
        )

    # Decode ONLY generated tokens — avoid duplication
    generated_ids  = outputs[0][inputs["input_ids"].shape[1]:]
    interpretation = gpt_tokenizer.decode(generated_ids, skip_special_tokens=True)

    return report_header + interpretation

# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("data/patient_text.csv")
    df["patient_id"] = df["ID"].str.rsplit("_", n=2).str[0]
    print(f"Dataset loaded: {df['patient_id'].nunique()} patients")

    # Test on 2 patients — one ADHD, one Control
    test_patients = df.groupby("label").first().reset_index()

    for _, row in test_patients.iterrows():
        pid = row["patient_id"] if "patient_id" in row else df[df["label"]==row["label"]]["patient_id"].iloc[0]
        pid = df[df["label"]==int(row["label"])]["patient_id"].iloc[0]

        print(f"\n{'='*50}")
        print(f"Testing patient: {pid}")

        classification, tbr, faa = classify_patient(pid, df)
        report = generate_clinical_report(pid, tbr, faa, classification)
        print(report)