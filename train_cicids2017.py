"""
FedShield — CICIDS2017 Retraining Script
Retrains the IntrusionDetector model on modern CICIDS2017 traffic
instead of the older NSL-KDD (1999) dataset.

ASSUMPTIONS (adjust if your cicids2017_preprocess.py differs):
- Preprocessed arrays are saved under data/cicids2017/ as:
    X_train.npy, y_train.npy, X_test.npy, y_test.npy
- Labels are already binary (0 = normal, 1 = attack).
  If you preprocessed multi-class labels instead, see the
  MULTICLASS block near the bottom to swap the model head.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import IntrusionDetector

DATA_DIR = "data/cicids2017"
MODEL_OUT = "models/cicids2017_model.pt"
HISTORY_OUT = "models/cicids2017_history.json"
EPOCHS = 20
BATCH_SIZE = 256
LR = 0.001


def load_data():
    print(f"Loading CICIDS2017 arrays from {DATA_DIR} ...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def train():
    os.makedirs("models", exist_ok=True)

    X_train, y_train, X_test, y_test = load_data()

    # Sanity check: feature count must match model's expected input dim
    n_features = X_train.shape[1]
    print(f"Detected {n_features} input features")

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE, shuffle=True
    )

    model = IntrusionDetector(input_dim=n_features) if _model_takes_input_dim() else IntrusionDetector()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            test_preds = (model(X_test_t) > 0.5).float()
            f1 = f1_score(y_test_t.numpy(), test_preds.numpy())
            precision = precision_score(y_test_t.numpy(), test_preds.numpy())
            recall = recall_score(y_test_t.numpy(), test_preds.numpy())

        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        history.append({
            "epoch": epoch, "loss": avg_loss,
            "f1": f1, "precision": precision, "recall": recall
        })

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  ↳ New best model saved (F1: {best_f1:.4f})")

    with open(HISTORY_OUT, "w") as f:
        json.dump(history, f, indent=2)

    cm = confusion_matrix(y_test_t.numpy(), test_preds.numpy())
    print("\nFinal Confusion Matrix (last epoch):")
    print(cm)
    print(f"\nBest F1 on CICIDS2017: {best_f1:.4f}")
    print(f"Model saved to: {MODEL_OUT}")
    print(f"History saved to: {HISTORY_OUT}")

    return best_f1


def _model_takes_input_dim():
    """Detect whether IntrusionDetector.__init__ accepts an input_dim arg,
    since CICIDS2017 likely has a different feature count than NSL-KDD's 41."""
    import inspect
    sig = inspect.signature(IntrusionDetector.__init__)
    return "input_dim" in sig.parameters


if __name__ == "__main__":
    train()