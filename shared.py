# synthfed/shared.py

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# ---- Model ----
class SimpleNet(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ---- Data Loading ----
def load_data(data_dir):

    features_file = data_dir / 'features' / 'idp' / 'freesurfer-7.3.2.tsv'
    labels_file = data_dir / 'labels' / 'labels.tsv'

    features_df = pd.read_csv(features_file, sep="\t")
    labels_df = pd.read_csv(labels_file, sep="\t")

    feature_cols = [col for col in features_df.columns if col not in ['participant_id', 'group']]
    X = features_df[feature_cols].values.astype(np.float32)

    # transform labels to integers
    label_mapping = {label: idx for idx, label in enumerate(labels_df['group'].unique())}
    y = labels_df['group'].map(label_mapping).values.astype(np.int64)

    return X, y

def make_loaders(data_dir, batch_size):
    X, y = load_data(data_dir)
    n = X.shape[0]
    split = int(n * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=batch_size)
    input_dim = X.shape[1]
    num_classes = int(y.max() + 1)
    return train_loader, val_loader, input_dim, num_classes

# ---- Required API ----
def get_initial_arrays(input_dim, dropout):
    num_classes = 2  # Adjust if known or pass as argument
    model = SimpleNet(input_dim, num_classes, dropout)
    arrays = [p.detach().cpu().numpy() for p in model.parameters()]
    return arrays

def train_local(arrays, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(config["data-path"])
    batch_size = int(config.get("batch-size", 32))
    epochs = int(config.get("local-epochs", 50))
    learning_rate = float(config.get("learning-rate", 0.0005))
    dropout = float(config.get("dropout", 0.0))
    train_loader, _, input_dim, num_classes = make_loaders(data_dir, batch_size)
    model = SimpleNet(input_dim, num_classes, dropout).to(device)
    with torch.no_grad():
        for p, arr in zip(model.parameters(), arrays):
            p.copy_(torch.tensor(arr))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    num_examples = 0
    for _ in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            num_examples += X.size(0)
    new_arrays = [p.detach().cpu().numpy() for p in model.parameters()]
    metrics = {"train_loss": float(loss.item())}
    return new_arrays, num_examples, metrics

def eval_local(arrays, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(config["data-path"])
    batch_size = int(config.get("batch-size", 32))
    dropout = float(config.get("dropout", 0.0))
    _, val_loader, input_dim, num_classes = make_loaders(data_dir, batch_size)
    model = SimpleNet(input_dim, num_classes, dropout).to(device)
    with torch.no_grad():
        for p, arr in zip(model.parameters(), arrays):
            p.copy_(torch.tensor(arr))
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            val_loss += loss.item() * X.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    avg_loss = val_loss / total if total > 0 else 0.0
    metrics = {
        "val_loss": avg_loss,
        "val_accuracy": correct / total if total > 0 else 0.0,
    }
    return avg_loss, total, metrics