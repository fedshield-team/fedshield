import argparse
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report
from server.aggregator import fed_avg

CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

# ---- Dataset selection ---------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", choices=["nslkdd", "cicids2017"], default="nslkdd",
    help="Which preprocessed dataset to run non-IID federated training on"
)
args = parser.parse_args()

if args.dataset == "cicids2017":
    X_TRAIN_PATH = "models/X_train_cicids2017.npy"
    Y_TRAIN_PATH = "models/y_train_cicids2017.npy"
    X_TEST_PATH  = "models/X_test_cicids2017.npy"
    Y_TEST_PATH  = "models/y_test_cicids2017.npy"
    OUT_MODEL    = "models/federated_noniid_model_cicids2017.pth"
    OUT_HISTORY  = "models/federated_noniid_history_cicids2017.json"
else:
    X_TRAIN_PATH = "data/X_train_mc.npy"
    Y_TRAIN_PATH = "data/y_train_mc.npy"
    X_TEST_PATH  = "data/X_test_mc.npy"
    Y_TEST_PATH  = "data/y_test_mc.npy"
    OUT_MODEL    = "models/federated_noniid_model.pth"
    OUT_HISTORY  = "models/federated_noniid_history.json"

print(f"===== DATASET: {args.dataset} =====")
# ---------------------------------------------------------------------------


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
    def __init__(self, node_id, X, y, label, lr=0.001):
        self.node_id = node_id
        self.label = label
        self.model = MultiClassIDS(input_dim=X.shape[1])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        X_t, y_t = torch.FloatTensor(X), torch.LongTensor(y)
        self.loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=True)
        self.X_t, self.y_t = X_t, y_t
        unique, counts = np.unique(y, return_counts=True)
        dist = {CLASS_NAMES[u]: c for u, c in zip(unique, counts)}
        print(f"[Node {node_id} - {label}] {len(X)} samples | Distribution: {dist}")

    def train_local(self, epochs=1):
        self.model.train()
        for _ in range(epochs):
            for X_b, y_b in self.loader:
                self.optimizer.zero_grad()
                self.criterion(self.model(X_b), y_b).backward()
                self.optimizer.step()

    def get_weights(self): return self.model.get_weights()
    def set_weights(self, w): self.model.set_weights(w)


# Load multi-class data (path depends on --dataset)
X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)
X_test  = np.load(X_TEST_PATH)
y_test  = np.load(Y_TEST_PATH)

n_features = X_train.shape[1]
print(f"Loaded {args.dataset}: {X_train.shape[0]} train rows, {n_features} features")

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

# Larger datasets (CICIDS2017) get more local gradient steps per round even with
# epochs=1, since each node still has hundreds of thousands of samples — so use a
# smaller learning rate to avoid client drift when averaging. NSL-KDD keeps the
# original, higher rate since its nodes are much smaller.
node_lr = 0.0003 if args.dataset == "cicids2017" else 0.001
local_epochs = 1 if args.dataset == "cicids2017" else 3

nodes = [
    MultiClassNode(1, X_train[hospital_idx], y_train[hospital_idx], "Hospital", lr=node_lr),
    MultiClassNode(2, X_train[bank_idx], y_train[bank_idx], "Bank", lr=node_lr),
    MultiClassNode(3, X_train[campus_idx], y_train[campus_idx], "Campus", lr=node_lr)
]

global_model = MultiClassIDS(input_dim=n_features)
history = []
ROUNDS = 15

print("\n===== FEDERATED TRAINING ON NON-IID DATA =====")
for round_num in range(1, ROUNDS+1):
    global_weights = global_model.get_weights()
    for node in nodes: node.set_weights(global_weights)
    for node in nodes: node.train_local(epochs=local_epochs)

    averaged = fed_avg([node.get_weights() for node in nodes])
    global_model.set_weights(averaged)

    global_model.eval()
    with torch.no_grad():
        preds = global_model(torch.FloatTensor(X_test)).argmax(dim=1)
        f1 = f1_score(y_test, preds.numpy(), average='macro')
    history.append({"round": round_num, "macro_f1": f1})
    print(f"Round {round_num:02d} | Global Macro F1: {f1:.4f}")

print(f"\n===== FINAL REPORT (NON-IID, {args.dataset}) =====")
global_model.eval()
with torch.no_grad():
    final_preds = global_model(torch.FloatTensor(X_test)).argmax(dim=1)
print(classification_report(y_test, final_preds.numpy(), target_names=CLASS_NAMES))

torch.save(global_model.state_dict(), OUT_MODEL)
with open(OUT_HISTORY, "w") as f:
    json.dump(history, f)
print(f"Non-IID federated model saved to {OUT_MODEL}")