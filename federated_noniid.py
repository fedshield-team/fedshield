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
    def __init__(self, node_id, X, y, label):
        self.node_id = node_id
        self.label = label
        self.model = MultiClassIDS()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        X_t, y_t = torch.FloatTensor(X), torch.LongTensor(y)
        self.loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=True)
        self.X_t, self.y_t = X_t, y_t
        unique, counts = np.unique(y, return_counts=True)
        dist = {CLASS_NAMES[u]: c for u, c in zip(unique, counts)}
        print(f"[Node {node_id} - {label}] {len(X)} samples | Distribution: {dist}")

    def train_local(self, epochs=3):
        self.model.train()
        for _ in range(epochs):
            for X_b, y_b in self.loader:
                self.optimizer.zero_grad()
                self.criterion(self.model(X_b), y_b).backward()
                self.optimizer.step()

    def get_weights(self): return self.model.get_weights()
    def set_weights(self, w): self.model.set_weights(w)


# Load multi-class data
X_train = np.load("data/X_train_mc.npy")
y_train = np.load("data/y_train_mc.npy")
X_test  = np.load("data/X_test_mc.npy")
y_test  = np.load("data/y_test_mc.npy")

print("===== CREATING NON-IID SPLIT =====")
print("Simulating: Hospital (mostly Normal+R2L), Bank (mostly DoS+Probe), Campus (mixed)\n")

# Non-IID split — each node sees a SKEWED distribution, like real organizations
np.random.seed(42)
idx_normal = np.where(y_train == 0)[0]
idx_dos    = np.where(y_train == 1)[0]
idx_probe  = np.where(y_train == 2)[0]
idx_r2l    = np.where(y_train == 3)[0]
idx_u2r    = np.where(y_train == 4)[0]

def take(idx, frac, seed_offset=0):
    n = int(len(idx) * frac)
    np.random.seed(42 + seed_offset)
    return np.random.choice(idx, n, replace=False)

# Hospital: mostly Normal traffic + most R2L (stealthy attacks targeting patient data)
hospital_idx = np.concatenate([
    take(idx_normal, 0.5, 1), take(idx_dos, 0.1, 2),
    take(idx_probe, 0.1, 3), take(idx_r2l, 0.6, 4), take(idx_u2r, 0.3, 5)
])

# Bank: heavy DoS/Probe target (financial systems get hammered with DDoS + scans)
bank_idx = np.concatenate([
    take(idx_normal, 0.25, 6), take(idx_dos, 0.6, 7),
    take(idx_probe, 0.6, 8), take(idx_r2l, 0.2, 9), take(idx_u2r, 0.3, 10)
])

# Campus: mixed, remaining data
used = set(hospital_idx) | set(bank_idx)
all_idx = set(range(len(y_train)))
campus_idx = np.array(list(all_idx - used))

nodes = [
    MultiClassNode(1, X_train[hospital_idx], y_train[hospital_idx], "Hospital"),
    MultiClassNode(2, X_train[bank_idx], y_train[bank_idx], "Bank"),
    MultiClassNode(3, X_train[campus_idx], y_train[campus_idx], "Campus")
]

global_model = MultiClassIDS()
history = []
ROUNDS = 15

print("\n===== FEDERATED TRAINING ON NON-IID DATA =====")
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

print("\n===== FINAL REPORT (NON-IID) =====")
global_model.eval()
with torch.no_grad():
    final_preds = global_model(torch.FloatTensor(X_test)).argmax(dim=1)
print(classification_report(y_test, final_preds.numpy(), target_names=CLASS_NAMES))

torch.save(global_model.state_dict(), "models/federated_noniid_model.pth")
with open("models/federated_noniid_history.json", "w") as f:
    json.dump(history, f)
print("Non-IID federated model saved!")