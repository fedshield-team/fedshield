import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import IntrusionDetector

class FedShieldClient(fl.client.NumPyClient):
    def __init__(self, node_id, X_train, y_train, X_test, y_test):
        self.node_id = node_id
        self.model = IntrusionDetector()
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        X_t = torch.FloatTensor(X_train)
        y_t = torch.FloatTensor(y_train).unsqueeze(1)
        self.loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=True)
        self.X_test = torch.FloatTensor(X_test)
        self.y_test = torch.FloatTensor(y_test)
        print(f"[Node {node_id}] Ready with {len(X_train)} samples")

    def get_parameters(self, config):
        return [p.detach().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, new_p in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new_p)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(3):
            for X_batch, y_batch in self.loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                self.optimizer.step()
        print(f"[Node {self.node_id}] Training round complete")
        return self.get_parameters(config), len(self.loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            preds = (self.model(self.X_test) > 0.5).float()
            f1 = f1_score(self.y_test.numpy(), preds.numpy())
            loss = self.criterion(
                self.model(self.X_test),
                self.y_test.unsqueeze(1)
            ).item()
        print(f"[Node {self.node_id}] F1: {f1:.4f}")
        return loss, len(self.X_test), {"f1": f1}


def start_client(node_id, server_address="127.0.0.1:8080"):
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")

    # Split data by node
    n = len(X_train)
    split = n // 3
    splits = {
        1: (X_train[:split], y_train[:split]),
        2: (X_train[split:2*split], y_train[split:2*split]),
        3: (X_train[2*split:], y_train[2*split:])
    }
    X, y = splits[node_id]

    client = FedShieldClient(node_id, X, y, X_test, y_test)
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client()
    )

if __name__ == "__main__":
    node_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    start_client(node_id)