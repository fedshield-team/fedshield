import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report
from server.aggregator import fed_avg

CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

class MultiClassIDS(nn.Module):
    def __init__(self, input_dim=41, num_classes=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.network(x)
    def get_weights(self): return [p.data.clone() for p in self.parameters()]
    def set_weights(self, w): 
        for p, w_ in zip(self.parameters(), w): p.data = w_.clone()

class MultiClassNode:
    def __init__(self, node_id, X, y):
        self.node_id = node_id
        self.model = MultiClassIDS()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        X_t, y_t = torch.FloatTensor(X), torch.LongTensor(y)
        self.loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=True)
        self.X_t, self.y_t = X_t, y_t
        print(f"[Node {node_id}] {len(X)} samples")

    def train_local(self, epochs=3):
        self.model.train()
        for _ in range(epochs):
            for X_b, y_b in self.loader:
                self.optimizer.zero_grad()
                self.criterion(self.model(X_b), y_b).backward()
                self.optimizer.step()

    def get_weights(self): return self.model.get_weights()
    def set_weights(self, w): self.model.set_weights(w)
    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.X_t).argmax(dim=1)
            return f1_score(self.y_t.numpy(), preds.numpy(), average='macro')

# Load multi-class data
X_train = np.load("data/X_train_mc.npy")
y_train = np.load("data/y_train_mc.npy")
X_test  = np.load("data/X_test_mc.npy")
y_test  = np.load("data/y_test_mc.npy")

# Split across 3 nodes
n = len(X_train) // 3
nodes = [
    MultiClassNode(1, X_train[:n], y_train[:n]),
    MultiClassNode(2, X_train[n:2*n], y_train[n:2*n]),
    MultiClassNode(3, X_train[2*n:], y_train[2*n:])
]

global_model = MultiClassIDS()
history = []
ROUNDS = 15

print("\n===== FEDERATED MULTI-CLASS TRAINING =====")
for round_num in range(1, ROUNDS+1):
    global_weights = global_model.get_weights()
    for node in nodes: node.set_weights(global_weights)
    for node in nodes: node.train_local(epochs=3)
    
    averaged = fed_avg([node.get_weights() for node in nodes])
    global_model.set_weights(averaged)
    
    global_model.eval()
    with torch.no_grad():
        preds = global_model(torch.FloatTensor(X_test)).argmax(dim=1)
        f1 = f1_score(y_test, preds.numpy(), average='macro')
    history.append({"round": round_num, "macro_f1": f1})
    print(f"Round {round_num:02d} | Global Macro F1: {f1:.4f}")

print("\n===== FINAL REPORT =====")
global_model.eval()
with torch.no_grad():
    final_preds = global_model(torch.FloatTensor(X_test)).argmax(dim=1)
print(classification_report(y_test, final_preds.numpy(), target_names=CLASS_NAMES))

torch.save(global_model.state_dict(), "models/federated_multiclass_model.pth")
with open("models/federated_multiclass_history.json", "w") as f:
    json.dump(history, f)
print("Federated multi-class model saved!")