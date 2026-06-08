from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Charger le modèle et le tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    num_labels=2  # ADHD ou non-ADHD
)

# Préparer un texte clinique
text = "Patient shows inattention, hyperactivity and impulsivity for 6 months."

# Tokenizer le texte
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

# Faire une prédiction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1)
    print(f"Prediction: {'ADHD' if prediction == 1 else 'No ADHD'}")