import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    f1_score
)
from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from server.aggregator import fed_avg


CLASS_NAMES = [
    "Normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R"
]


class MultiClassIDS(nn.Module):

    def __init__(
        self,
        input_dim=41,
        num_classes=5
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

    def get_weights(self):

        return [
            parameter.detach().clone()
            for parameter in self.parameters()
        ]

    def set_weights(self, weights):

        with torch.no_grad():

            for parameter, weight in zip(
                self.parameters(),
                weights
            ):

                parameter.copy_(weight)


class MultiClassNode:

    def __init__(
        self,
        node_id,
        X,
        y,
        label,
        lr=0.001
    ):

        self.node_id = node_id
        self.label = label
        self.lr = lr

        self.model = MultiClassIDS(
            input_dim=X.shape[1]
        )

        self.criterion = (
            nn.CrossEntropyLoss()
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        X_t = torch.as_tensor(
            X,
            dtype=torch.float32
        )

        y_t = torch.as_tensor(
            y,
            dtype=torch.long
        )

        self.X_t = X_t
        self.y_t = y_t

        self.loader = DataLoader(
            TensorDataset(
                X_t,
                y_t
            ),
            batch_size=256,
            shuffle=True
        )

        unique, counts = np.unique(
            y,
            return_counts=True
        )

        distribution = {
            CLASS_NAMES[int(label)]:
                int(count)
            for label, count
            in zip(unique, counts)
        }

        print(
            f"[Node {node_id} - {label}] "
            f"{len(X)} samples | "
            f"Distribution: {distribution}"
        )

    def train_local(
        self,
        epochs=1
    ):

        self.model.train()

        total_loss = 0.0
        batches = 0

        for _ in range(epochs):

            for X_batch, y_batch in self.loader:

                self.optimizer.zero_grad()

                logits = self.model(
                    X_batch
                )

                loss = self.criterion(
                    logits,
                    y_batch
                )

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batches += 1

        return total_loss / max(
            batches,
            1
        )

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):

        self.model.set_weights(
            weights
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    choices=[
        "nslkdd",
        "cicids2017"
    ],
    default="nslkdd"
)

args = parser.parse_args()


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


if args.dataset == "cicids2017":

    X_TRAIN_PATH = os.path.join(
        MODELS_DIR,
        "X_train_cicids2017.npy"
    )

    Y_TRAIN_PATH = os.path.join(
        MODELS_DIR,
        "y_train_cicids2017.npy"
    )

    X_TEST_PATH = os.path.join(
        MODELS_DIR,
        "X_test_cicids2017.npy"
    )

    Y_TEST_PATH = os.path.join(
        MODELS_DIR,
        "y_test_cicids2017.npy"
    )

    OUT_MODEL = os.path.join(
        MODELS_DIR,
        "federated_noniid_model_cicids2017.pth"
    )

    OUT_HISTORY = os.path.join(
        MODELS_DIR,
        "federated_noniid_history_cicids2017.json"
    )

else:

    X_TRAIN_PATH = os.path.join(
        DATA_DIR,
        "X_train_mc.npy"
    )

    Y_TRAIN_PATH = os.path.join(
        DATA_DIR,
        "y_train_mc.npy"
    )

    X_TEST_PATH = os.path.join(
        DATA_DIR,
        "X_test_mc.npy"
    )

    Y_TEST_PATH = os.path.join(
        DATA_DIR,
        "y_test_mc.npy"
    )

    OUT_MODEL = os.path.join(
        MODELS_DIR,
        "federated_noniid_model.pth"
    )

    OUT_HISTORY = os.path.join(
        MODELS_DIR,
        "federated_noniid_history.json"
    )


print(
    f"\n===== DATASET: {args.dataset} ====="
)


X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)


input_dim = X_train.shape[1]

print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Features: {input_dim}"
)


# ============================================================
# NON-IID SPLIT
# ============================================================

np.random.seed(42)

idx_normal = np.where(
    y_train == 0
)[0]

idx_dos = np.where(
    y_train == 1
)[0]

idx_probe = np.where(
    y_train == 2
)[0]

idx_r2l = np.where(
    y_train == 3
)[0]

idx_u2r = np.where(
    y_train == 4
)[0]


def take(
    indices,
    fraction,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    count = int(
        len(indices) * fraction
    )

    count = min(
        count,
        len(indices)
    )

    return rng.choice(
        indices,
        count,
        replace=False
    )


hospital_idx = np.concatenate(
    [

        take(
            idx_normal,
            0.50,
            1
        ),

        take(
            idx_dos,
            0.10,
            2
        ),

        take(
            idx_probe,
            0.10,
            3
        ),

        take(
            idx_r2l,
            0.60,
            4
        ),

        take(
            idx_u2r,
            0.30,
            5
        )
    ]
)


bank_idx = np.concatenate(
    [

        take(
            idx_normal,
            0.25,
            6
        ),

        take(
            idx_dos,
            0.60,
            7
        ),

        take(
            idx_probe,
            0.60,
            8
        ),

        take(
            idx_r2l,
            0.20,
            9
        ),

        take(
            idx_u2r,
            0.30,
            10
        )
    ]
)


used = set(
    hospital_idx.tolist()
) | set(
    bank_idx.tolist()
)

all_indices = set(
    range(len(y_train))
)

campus_idx = np.array(
    list(
        all_indices - used
    ),
    dtype=int
)


print(
    "\n===== NON-IID NODE DISTRIBUTION ====="
)


nodes = [

    MultiClassNode(
        1,
        X_train[hospital_idx],
        y_train[hospital_idx],
        "Hospital",
        lr=(
            0.0003
            if args.dataset == "cicids2017"
            else 0.001
        )
    ),

    MultiClassNode(
        2,
        X_train[bank_idx],
        y_train[bank_idx],
        "Bank",
        lr=(
            0.0003
            if args.dataset == "cicids2017"
            else 0.001
        )
    ),

    MultiClassNode(
        3,
        X_train[campus_idx],
        y_train[campus_idx],
        "Campus",
        lr=(
            0.0003
            if args.dataset == "cicids2017"
            else 0.001
        )
    )
]


sample_counts = [
    len(node.X_t)
    for node in nodes
]


local_epochs = (
    1
    if args.dataset == "cicids2017"
    else 3
)


global_model = MultiClassIDS(
    input_dim=input_dim
)


X_test_t = torch.as_tensor(
    X_test,
    dtype=torch.float32
)


history = []

ROUNDS = 15


print(
    "\n===== FEDERATED NON-IID TRAINING ====="
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

    losses = []

    for node in nodes:

        loss = node.train_local(
            epochs=local_epochs
        )

        losses.append(loss)

    averaged = fed_avg(
        [
            node.get_weights()
            for node in nodes
        ],
        sample_counts
    )

    global_model.set_weights(
        averaged
    )

    global_model.eval()

    with torch.no_grad():

        predictions = (
            global_model(
                X_test_t
            )
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    avg_loss = float(
        np.average(
            losses,
            weights=sample_counts
        )
    )

    history.append(
        {
            "round": round_num,
            "loss": avg_loss,
            "macro_f1": float(macro_f1)
        }
    )

    print(
        f"Global Macro F1: "
        f"{macro_f1:.4f}"
    )


print(
    f"\n===== FINAL REPORT "
    f"(NON-IID, {args.dataset}) ====="
)


global_model.eval()

with torch.no_grad():

    final_predictions = (
        global_model(X_test_t)
        .argmax(dim=1)
        .cpu()
        .numpy()
    )


print(
    classification_report(
        y_test,
        final_predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


torch.save(
    global_model.state_dict(),
    OUT_MODEL
)

with open(
    OUT_HISTORY,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history,
        f,
        indent=2
    )


print(
    f"Model saved to {OUT_MODEL}"
)

print(
    f"History saved to {OUT_HISTORY}"
)