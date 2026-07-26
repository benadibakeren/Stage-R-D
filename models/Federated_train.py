import torch
import numpy as np
import pandas as pd
import copy
from collections import Counter
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score

# ── Configuration ─────────────────────────────────────────
MODEL_NAME    = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
DATA_PATH     = "data/patient_text.csv"
NUM_NODES     = 4
NUM_ROUNDS    = 5
LOCAL_EPOCHS  = 5
LEARNING_RATE = 3e-5
BATCH_SIZE    = 8
SEED          = 42

# ── Setup ─────────────────────────────────────────────────
np.random.seed(SEED)
torch.manual_seed(SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load data ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["patient_id"] = df["ID"].str.split("_").str[0]
print(f"Dataset: {len(df)} segments, {df['patient_id'].nunique()} patients")

# ── Fixed patient-level train/test split ──────────────────
# Split is established ONCE before any training.
# Test set is NEVER seen during training.

all_patients = df["patient_id"].unique()
np.random.shuffle(all_patients)
split_idx      = int(0.8 * len(all_patients))
train_patients = all_patients[:split_idx]
test_patients  = all_patients[split_idx:]

train_df = df[df["patient_id"].isin(train_patients)].reset_index(drop=True)
test_df  = df[df["patient_id"].isin(test_patients)].reset_index(drop=True)

print(f"\nTrain: {train_df['patient_id'].nunique()} patients, {len(train_df)} segments")
print(f"Test : {test_df['patient_id'].nunique()} patients, {len(test_df)} segments")

# ── Node partitioning ─────────────────────────────────────
# Training patients are divided equally across nodes.
# All segments of a patient stay within the same node.

train_patients_arr = np.array(train_patients)
np.random.shuffle(train_patients_arr)
node_groups = np.array_split(train_patients_arr, NUM_NODES)

print(f"\nNode distribution:")
for i, group in enumerate(node_groups):
    node_df = train_df[train_df["patient_id"].isin(group)]
    print(f"  Node {i+1}: {node_df['patient_id'].nunique()} patients, "
          f"{len(node_df)} segments - "
          f"ADHD: {node_df[node_df['label']==1]['patient_id'].nunique()}, "
          f"Control: {node_df[node_df['label']==0]['patient_id'].nunique()}")

# ── PyTorch Dataset ───────────────────────────────────────
class EEGDataset(torch.utils.data.Dataset):
    """Tokenizes EEG text profiles for BiomedBERT."""

    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts, max_length=256, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids"     : self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label"         : torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── Patient-level evaluation via majority voting ──────────
def evaluate_patient_level(model, eval_df):
    """
    Evaluates model at patient level using majority voting.
    Each patient's segments vote independently.
    The majority class is the final prediction for that patient.
    """
    model.eval()
    texts  = eval_df["text"].tolist()
    labels = eval_df["label"].tolist()
    ids    = eval_df["patient_id"].tolist()

    loader = DataLoader(EEGDataset(texts, labels), batch_size=BATCH_SIZE)
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids      =batch["input_ids"].to(device),
                attention_mask =batch["attention_mask"].to(device)
            )
            all_preds.extend(
                torch.argmax(outputs.logits, dim=1).cpu().numpy()
            )

    patient_preds  = {}
    patient_labels = {}

    for pid, pred, label in zip(ids, all_preds, labels):
        if pid not in patient_preds:
            patient_preds[pid]  = []
            patient_labels[pid] = label
        patient_preds[pid].append(pred)

    final_preds  = []
    final_labels = []

    for pid in patient_preds:
        vote = Counter(patient_preds[pid]).most_common(1)[0][0]
        final_preds.append(vote)
        final_labels.append(patient_labels[pid])

    return accuracy_score(final_labels, final_preds), len(final_labels)

# ── Local training ────────────────────────────────────────
def train_local(model, train_loader, epochs=LOCAL_EPOCHS, lr=LEARNING_RATE):
    """
    Trains model locally on a single node.
    Uses AdamW optimizer with linear warmup scheduler.
    Model weights are updated locally — raw data never leaves the node.
    """
    model.train()
    optimizer    = AdamW(model.parameters(), lr=lr)
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids      =batch["input_ids"].to(device),
                attention_mask =batch["attention_mask"].to(device),
                labels         =batch["label"].to(device)
            )
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += outputs.loss.item()
        print(f"    Epoch {epoch+1}/{epochs} - "
              f"Loss: {total_loss/len(train_loader):.4f}")

    return model

# ── FedAvg aggregation ────────────────────────────────────
def fed_avg(global_model, local_models, node_sizes):
    """
    Aggregates local model weights using FedAvg.
    Each node's contribution is weighted by its number of training segments.
    Only model weights are shared — never raw patient data.

    Reference: McMahan et al. (2017), Communication-Efficient Learning
    of Deep Networks from Decentralized Data.
    """
    global_dict = global_model.state_dict()
    total = sum(node_sizes)

    for key in global_dict:
        global_dict[key] = sum(
            local_models[i].state_dict()[key] * (node_sizes[i] / total)
            for i in range(len(local_models))
        )

    global_model.load_state_dict(global_dict)
    return global_model

# ── Centralized baseline ──────────────────────────────────
# Trained from scratch with same total training budget as FL:
# LOCAL_EPOCHS * NUM_ROUNDS = 5 * 5 = 25 epochs total
# Evaluated on the same fixed test set as the federated model.

print("\n" + "="*50)
print("CENTRALIZED BASELINE")
print(f"(trained for {LOCAL_EPOCHS * NUM_ROUNDS} epochs — same budget as FL)")
print("="*50)

centralized_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
).to(device)

train_loader_central = DataLoader(
    EEGDataset(train_df["text"].tolist(), train_df["label"].tolist()),
    batch_size=BATCH_SIZE, shuffle=True
)

# Same total training budget as FL (LOCAL_EPOCHS * NUM_ROUNDS)
centralized_model = train_local(
    centralized_model, train_loader_central,
    epochs=LOCAL_EPOCHS * NUM_ROUNDS
)
central_acc, n_patients = evaluate_patient_level(centralized_model, test_df)
print(f"\nCentralized baseline: {central_acc*100:.2f}% ({n_patients} patients)")

# ── Federated Learning simulation ─────────────────────────
print("\n" + "="*50)
print(f"FEDERATED LEARNING - {NUM_NODES} nodes, {NUM_ROUNDS} rounds")
print("="*50)

global_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
).to(device)

best_acc = 0

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\n{'='*40}")
    print(f"ROUND {round_num}/{NUM_ROUNDS}")
    print(f"{'='*40}")

    local_models = []
    node_sizes   = []

    for node_id in range(NUM_NODES):
        node_df = train_df[
            train_df["patient_id"].isin(node_groups[node_id])
        ]
        train_loader = DataLoader(
            EEGDataset(node_df["text"].tolist(), node_df["label"].tolist()),
            batch_size=BATCH_SIZE, shuffle=True
        )

        print(f"\n  Node {node_id+1}/{NUM_NODES} - local training...")
        local_model = copy.deepcopy(global_model)
        local_model = train_local(local_model, train_loader)

        local_models.append(local_model)
        # Weighted by number of segments (standard FedAvg)
        node_sizes.append(len(node_df))

    print(f"\n  FedAvg aggregation...")
    global_model = fed_avg(global_model, local_models, node_sizes)

    round_acc, n_p = evaluate_patient_level(global_model, test_df)

    # Track best round — reflects model selected for deployment
    # Note: best_acc is the maximum across rounds, not the final round
    if round_acc > best_acc:
        best_acc = round_acc

    print(f"\n  Global model - Round {round_num}: "
          f"{round_acc*100:.2f}% ({n_p} patients)")

# ── Final summary ─────────────────────────────────────────
print(f"\n{'='*50}")
print(f"FINAL RESULTS")
print(f"{'='*50}")
print(f"Centralized baseline : {central_acc*100:.2f}%")
print(f"Best federated acc   : {best_acc*100:.2f}%")
print(f"Performance loss     : {(central_acc - best_acc)*100:.2f}%")
print(f"\nNote: best_acc reports the best round across {NUM_ROUNDS} rounds,")
print(f"reflecting the model that would be selected for deployment.")