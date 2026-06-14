import numpy as np
import json
import torch
from nodes.node import FedNode
from server.aggregator import fed_avg
from model import IntrusionDetector
from sklearn.metrics import f1_score, classification_report
import torch

# Load preprocessed data
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# Split data across 3 nodes — simulating 3 organizations
# Each node sees different data, never shares it
n = len(X_train)
split = n // 3

node1 = FedNode(1, X_train[:split], y_train[:split])
node2 = FedNode(2, X_train[split:2*split], y_train[split:2*split])
node3 = FedNode(3, X_train[2*split:], y_train[2*split:])
nodes = [node1, node2, node3]

# Global model (lives on server)
global_model = IntrusionDetector()

print("\n========== FEDERATED LEARNING STARTED ==========")
print(f"Nodes: 3 | Rounds: 15 | Local epochs per round: 3\n")

history = []
ROUNDS = 15

for round_num in range(1, ROUNDS + 1):
    print(f"--- Round {round_num}/{ROUNDS} ---")
    
    # Step 1: Send global weights to all nodes
    global_weights = [p.data.clone() for p in global_model.parameters()]
    for node in nodes:
        node.set_weights(global_weights)
    
    # Step 2: Each node trains locally
    for node in nodes:
        node.train_local(epochs=3)
    
    # Step 3: Collect weights from all nodes
    all_weights = [node.get_weights() for node in nodes]
    
    # Step 4: FedAvg aggregation on server
    averaged_weights = fed_avg(all_weights)
    
    # Step 5: Update global model
    for p, w in zip(global_model.parameters(), averaged_weights):
        p.data = w.clone()
    
    # Step 6: Evaluate global model on test set
    global_model.eval()
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    with torch.no_grad():
        preds = (global_model(X_test_t) > 0.5).float()
        f1 = f1_score(y_test_t.numpy(), preds.numpy())
    
    history.append({"round": round_num, "f1": f1})
    print(f"Global Model F1 after Round {round_num}: {f1:.4f}\n")

# Final evaluation
print("========== FINAL RESULTS ==========")
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)
global_model.eval()
with torch.no_grad():
    final_preds = (global_model(X_test_t) > 0.5).float()
print(classification_report(y_test_t.numpy(), final_preds.numpy(),
                             target_names=["Normal", "Attack"]))

# Save
torch.save(global_model.state_dict(), "models/federated_model.pth")
with open("models/federated_history.json", "w") as f:
    json.dump(history, f)

print(f"Baseline F1:  0.9947")
print(f"Federated F1: {history[-1]['f1']:.4f}")
print("Federated model saved!")