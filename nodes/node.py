import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from model import IntrusionDetector


class FedNode:

    def __init__(self, node_id, X, y, input_dim=None, lr=0.001):
        self.node_id = node_id

        if input_dim is None:
            input_dim = X.shape[1]

        self.model = IntrusionDetector(input_dim=input_dim)

        self.criterion = nn.BCEWithLogitsLoss()

        self.lr = lr
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(
            y,
            dtype=torch.float32
        ).view(-1, 1)

        dataset = TensorDataset(X_t, y_t)

        self.loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=True
        )

        self.X_t = X_t
        self.y_t = y_t

        print(
            f"[Node {node_id}] "
            f"Initialized with {len(X)} samples | "
            f"Features: {input_dim}"
        )

    def train_local(self, epochs=3):

        self.model.train()

        total_loss = 0.0
        batches = 0

        for _ in range(epochs):

            for X_batch, y_batch in self.loader:

                self.optimizer.zero_grad()

                logits = self.model(X_batch)

                loss = self.criterion(
                    logits,
                    y_batch
                )

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batches += 1

        avg_loss = total_loss / max(batches, 1)

        print(
            f"[Node {self.node_id}] "
            f"Local training complete | "
            f"Loss: {avg_loss:.4f}"
        )

        return avg_loss

    def get_weights(self):

        return [
            parameter.detach().clone()
            for parameter in self.model.parameters()
        ]

    def set_weights(self, weights):

        self.model.set_weights(weights)

        # Important:
        # Local optimizer state should not survive a global model reset.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

    def evaluate(self):

        self.model.eval()

        with torch.no_grad():

            probabilities = torch.sigmoid(
                self.model(self.X_t)
            )

            predictions = (
                probabilities >= 0.5
            ).long().view(-1)

        y_true = self.y_t.long().view(-1)

        f1 = f1_score(
            y_true.numpy(),
            predictions.numpy(),
            zero_division=0
        )

        print(
            f"[Node {self.node_id}] "
            f"F1: {f1:.4f}"
        )

        return f1