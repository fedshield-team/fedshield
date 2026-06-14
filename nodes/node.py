import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import IntrusionDetector

class FedNode:
    def __init__(self, node_id, X, y):
        self.node_id = node_id
        self.model = IntrusionDetector()
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y).unsqueeze(1)
        dataset = TensorDataset(X_t, y_t)
        self.loader = DataLoader(dataset, batch_size=256, shuffle=True)
        self.X_t = X_t
        self.y_t = y_t
        print(f"[Node {node_id}] Initialized with {len(X)} samples")

    def train_local(self, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in self.loader:
                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        avg_loss = total_loss / len(self.loader)
        print(f"[Node {self.node_id}] Local training done | Loss: {avg_loss:.4f}")
        return avg_loss

    def get_weights(self):
        return [p.data.clone() for p in self.model.parameters()]

    def set_weights(self, weights):
        for p, w in zip(self.model.parameters(), weights):
            p.data = w.clone()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            preds = (self.model(self.X_t) > 0.5).float()
            f1 = f1_score(self.y_t.numpy(), preds.numpy())
        print(f"[Node {self.node_id}] F1: {f1:.4f}")
        return f1