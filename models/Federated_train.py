import flwr as fl
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from flwr.client import NumPyClient, ClientApp
from flwr.server import ServerApp, ServerConfig
from flwr.simulation import run_simulation

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv('data/patient_text.csv')

# ── Charger et partitionner les données ──────────────────
df["patient_id"] = df["ID"].str.split("_").str[0]
patients = df["patient_id"].unique()
np.random.seed(42)
np.random.shuffle(patients)

# 4 nœuds — chaque nœud a ses patients
nodes = np.array_split(patients, 4)

def get_node_data(node_id):
    node_patients = nodes[node_id]
    node_df = df[df["patient_id"].isin(node_patients)]
    texts = node_df["text"].tolist()
    labels = node_df["label"].tolist()
    split = int(0.8 * len(texts))
    return texts[:split], labels[:split], texts[split:], labels[split:]

# ── Dataset PyTorch ───────────────────────────────────────
class EEGDataset(torch.utils.data.Dataset):
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
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── Client Flower ─────────────────────────────────────────
class ADHDClient(NumPyClient):
    def __init__(self, node_id):
        self.node_id = node_id
        train_texts, train_labels, val_texts, val_labels = get_node_data(node_id)
        self.train_loader = DataLoader(EEGDataset(train_texts, train_labels), batch_size=8, shuffle=True)
        self.val_loader = DataLoader(EEGDataset(val_texts, val_labels), batch_size=8)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=3e-5)
        for batch in self.train_loader:
            optimizer.zero_grad()
            outputs = self.model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["label"].to(device)
            )
            outputs.loss.backward()
            optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        preds, true = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                outputs = self.model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device)
                )
                preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                true.extend(batch["label"].numpy())
        acc = accuracy_score(true, preds)
        print(f"Node {self.node_id} accuracy: {acc*100:.2f}%")
        return 0.0, len(self.val_loader.dataset), {"accuracy": acc}

# ── Lancer la simulation ──────────────────────────────────
from flwr.server.strategy import FedAvg
from flwr.common import Context
from flwr.server import ServerAppComponents

def server_fn(context: Context):
    strategy = FedAvg()
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)

def client_fn(context: Context):
    node_id = int(context.node_id) % 4
    return ADHDClient(node_id).to_client()

client = ClientApp(client_fn=client_fn)
server = ServerApp(server_fn=server_fn)

run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=4,
)