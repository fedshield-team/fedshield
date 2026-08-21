import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    f1_score
)

from model import IntrusionDetector
from nodes.node import FedNode
from server.aggregator import fed_avg


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


X_train = np.load(
    os.path.join(
        DATA_DIR,
        "X_train.npy"
    )
)

y_train = np.load(
    os.path.join(
        DATA_DIR,
        "y_train.npy"
    )
)

X_test = np.load(
    os.path.join(
        DATA_DIR,
        "X_test.npy"
    )
)

y_test = np.load(
    os.path.join(
        DATA_DIR,
        "y_test.npy"
    )
)


input_dim = X_train.shape[1]

print(
    f"Training features: {input_dim}"
)

n = len(X_train)

split = n // 3


node1 = FedNode(
    1,
    X_train[:split],
    y_train[:split],
    input_dim=input_dim
)

node2 = FedNode(
    2,
    X_train[split:2 * split],
    y_train[split:2 * split],
    input_dim=input_dim
)

node3 = FedNode(
    3,
    X_train[2 * split:],
    y_train[2 * split:],
    input_dim=input_dim
)

nodes = [
    node1,
    node2,
    node3
]


sample_counts = [
    len(node1.X_t),
    len(node2.X_t),
    len(node3.X_t)
]


global_model = IntrusionDetector(
    input_dim=input_dim
)

history = []

ROUNDS = 15
LOCAL_EPOCHS = 3


print(
    "\n========== FEDERATED LEARNING =========="
)

print(
    f"Nodes: {len(nodes)}"
)

print(
    f"Rounds: {ROUNDS}"
)

print(
    f"Local epochs: {LOCAL_EPOCHS}"
)


X_test_t = torch.as_tensor(
    X_test,
    dtype=torch.float32
)


for round_num in range(
    1,
    ROUNDS + 1
):

    print(
        f"\n--- Round "
        f"{round_num}/{ROUNDS} ---"
    )

    global_weights = (
        global_model.get_weights()
    )

    for node in nodes:

        node.set_weights(
            global_weights
        )

    local_losses = []

    for node in nodes:

        loss = node.train_local(
            epochs=LOCAL_EPOCHS
        )

        local_losses.append(loss)

    client_weights = [
        node.get_weights()
        for node in nodes
    ]

    averaged_weights = fed_avg(
        client_weights,
        sample_counts
    )

    global_model.set_weights(
        averaged_weights
    )

    global_model.eval()

    with torch.no_grad():

        probabilities = torch.sigmoid(
            global_model(X_test_t)
        )

        predictions = (
            probabilities >= 0.5
        ).long().view(-1)

    f1 = f1_score(
        y_test,
        predictions.numpy(),
        zero_division=0
    )

    avg_loss = float(
        np.average(
            local_losses,
            weights=sample_counts
        )
    )

    history.append(
        {
            "round": round_num,
            "loss": avg_loss,
            "f1": float(f1)
        }
    )

    print(
        f"Global F1: {f1:.4f} | "
        f"Average local loss: {avg_loss:.4f}"
    )


print(
    "\n========== FINAL RESULTS =========="
)


global_model.eval()

with torch.no_grad():

    probabilities = torch.sigmoid(
        global_model(X_test_t)
    )

    final_predictions = (
        probabilities >= 0.5
    ).long().view(-1)


print(
    classification_report(
        y_test,
        final_predictions.numpy(),
        target_names=[
            "Normal",
            "Attack"
        ],
        zero_division=0
    )
)


model_path = os.path.join(
    MODELS_DIR,
    "federated_model.pth"
)

history_path = os.path.join(
    MODELS_DIR,
    "federated_history.json"
)


torch.save(
    global_model.state_dict(),
    model_path
)

with open(
    history_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history,
        f,
        indent=2
    )


print(
    f"Federated model saved: {model_path}"
)

print(
    f"History saved: {history_path}"
)