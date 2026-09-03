"""Flower client for the authoritative FedShield multiclass model path.

The client intentionally consumes the already-preprocessed multiclass arrays.
The scaler and categorical encoders are loaded and validated before a client
starts so that a Flower run cannot silently fall back to the legacy binary
dataset or an incompatible preprocessing contract.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import flwr as fl
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from model import (
    MULTICLASS_INPUT_DIM,
    MULTICLASS_NUM_CLASSES,
    MultiClassIDS,
)


REQUIRED_ENCODERS = {"protocol_type", "service", "flag"}


def _validate_multiclass_contract(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
    encoders,
) -> None:
    """Reject data or preprocessors that cannot serve MultiClassIDS."""
    if X_train.ndim != 2 or X_train.shape[1] != MULTICLASS_INPUT_DIM:
        raise ValueError(
            "Multiclass training data must have exactly "
            f"{MULTICLASS_INPUT_DIM} features; got {X_train.shape}"
        )
    if X_test.ndim != 2 or X_test.shape[1] != MULTICLASS_INPUT_DIM:
        raise ValueError(
            "Multiclass test data must have exactly "
            f"{MULTICLASS_INPUT_DIM} features; got {X_test.shape}"
        )
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Multiclass features and labels must have matching lengths")

    labels = np.concatenate((np.asarray(y_train), np.asarray(y_test)))
    if labels.size and (
        labels.min() < 0 or labels.max() >= MULTICLASS_NUM_CLASSES
    ):
        raise ValueError(
            f"Multiclass labels must be in [0, {MULTICLASS_NUM_CLASSES - 1}]"
        )

    scaler_features = getattr(scaler, "n_features_in_", MULTICLASS_INPUT_DIM)
    if scaler_features != MULTICLASS_INPUT_DIM:
        raise ValueError(
            "Multiclass scaler must be fitted for "
            f"{MULTICLASS_INPUT_DIM} features; got {scaler_features}"
        )
    if not isinstance(encoders, dict) or not REQUIRED_ENCODERS.issubset(
        encoders.keys()
    ):
        missing = REQUIRED_ENCODERS - set(encoders.keys())
        raise ValueError(
            "Missing multiclass categorical encoders: "
            + ", ".join(sorted(missing))
        )


def load_multiclass_data(base_dir: str | os.PathLike[str] | None = None):
    """Load the real multiclass arrays and preprocessing artifacts."""
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[1]
    data_dir = base / "data"
    models_dir = base / "models"

    X_train = np.load(data_dir / "X_train_mc.npy")
    y_train = np.load(data_dir / "y_train_mc.npy")
    X_test = np.load(data_dir / "X_test_mc.npy")
    y_test = np.load(data_dir / "y_test_mc.npy")
    scaler = joblib.load(models_dir / "scaler_multiclass.pkl")
    encoders = joblib.load(models_dir / "encoders_multiclass.pkl")

    _validate_multiclass_contract(
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
        encoders,
    )
    return X_train, y_train, X_test, y_test


class FedShieldClient(fl.client.NumPyClient):
    """A Flower client training MultiClassIDS on one local data partition."""

    def __init__(
        self,
        node_id,
        X_train,
        y_train,
        X_test,
        y_test,
        input_dim=MULTICLASS_INPUT_DIM,
        num_classes=MULTICLASS_NUM_CLASSES,
        lr=0.001,
        local_epochs=3,
    ):
        if input_dim != MULTICLASS_INPUT_DIM:
            raise ValueError(
                f"FedShield multiclass clients require {MULTICLASS_INPUT_DIM} features"
            )
        if num_classes != MULTICLASS_NUM_CLASSES:
            raise ValueError(
                f"FedShield multiclass clients require {MULTICLASS_NUM_CLASSES} classes"
            )

        self.node_id = node_id
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr
        self.local_epochs = local_epochs
        self.model = MultiClassIDS(
            input_dim=MULTICLASS_INPUT_DIM,
            num_classes=MULTICLASS_NUM_CLASSES,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
        y_train_t = torch.as_tensor(y_train, dtype=torch.long)
        self.loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=256,
            shuffle=True,
        )
        self.X_test = torch.as_tensor(X_test, dtype=torch.float32)
        self.y_test = torch.as_tensor(y_test, dtype=torch.long)

        print(
            f"[Node {node_id}] Ready | "
            f"Train: {len(X_train)} | "
            f"Features: {self.input_dim} | "
            f"Classes: {self.num_classes}"
        )

    def get_parameters(self, config):
        return [
            parameter.detach().cpu().numpy()
            for parameter in self.model.parameters()
        ]

    def set_parameters(self, parameters):
        model_parameters = list(self.model.parameters())
        if len(parameters) != len(model_parameters):
            raise ValueError(
                "Received incorrect number of model tensors for MultiClassIDS"
            )

        weights = []
        for parameter, new_parameter in zip(model_parameters, parameters):
            new_tensor = torch.as_tensor(new_parameter, dtype=parameter.dtype)
            if parameter.shape != new_tensor.shape:
                raise ValueError(
                    "Received incompatible MultiClassIDS tensor shape: "
                    f"{tuple(new_tensor.shape)} != {tuple(parameter.shape)}"
                )
            weights.append(new_tensor)
        self.model.set_weights(weights)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )
        self.model.train()

        total_loss = 0.0
        batches = 0
        for _ in range(self.local_epochs):
            for X_batch, y_batch in self.loader:
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=5.0,
                )
                self.optimizer.step()
                total_loss += loss.item()
                batches += 1

        avg_loss = total_loss / max(batches, 1)
        print(
            f"[Node {self.node_id}] Training complete | "
            f"Loss: {avg_loss:.4f}"
        )
        return (
            self.get_parameters(config),
            len(self.loader.dataset),
            {"loss": float(avg_loss)},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X_test)
            loss = self.criterion(logits, self.y_test).item()
            predictions = logits.argmax(dim=1)

        macro_f1 = f1_score(
            self.y_test.numpy(),
            predictions.numpy(),
            average="macro",
            zero_division=0,
        )
        print(
            f"[Node {self.node_id}] Evaluation macro F1: {macro_f1:.4f}"
        )
        return (
            float(loss),
            len(self.X_test),
            {"macro_f1": float(macro_f1)},
        )


def start_client(node_id, server_address=None):
    if server_address is None:
        server_address = os.environ.get("SERVER_ADDRESS", "server:8080")

    X_train, y_train, X_test, y_test = load_multiclass_data()
    n = len(X_train)
    split = n // 3
    splits = {
        1: (X_train[:split], y_train[:split]),
        2: (X_train[split : 2 * split], y_train[split : 2 * split]),
        3: (X_train[2 * split :], y_train[2 * split :]),
    }
    if node_id not in splits:
        raise ValueError("node_id must be 1, 2, or 3")

    X, y = splits[node_id]
    client = FedShieldClient(
        node_id=node_id,
        X_train=X,
        y_train=y,
        X_test=X_test,
        y_test=y_test,
    )
    print(
        f"[Node {node_id}] Connecting to Flower server at {server_address}"
    )
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    node_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    start_client(node_id)