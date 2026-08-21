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

            nn.Linear(
                input_dim,
                256
            ),

            nn.LayerNorm(256),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(
                256,
                128
            ),

            nn.LayerNorm(128),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):
        return self.network(x)

    def get_weights(self):

        return [
            parameter.detach().clone()
            for parameter in self.parameters()
        ]

    def set_weights(self, weights):

        if len(weights) != len(
            list(self.parameters())
        ):
            raise ValueError(
                "Incorrect number of model tensors"
            )

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
        lr=0.001
    ):

        self.node_id = node_id
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
            f"[Node {node_id}] "
            f"{len(X)} samples | "
            f"{distribution}"
        )

    def train_local(
        self,
        epochs=3
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
        "X_train_mc.npy"
    )
)

y_train = np.load(
    os.path.join(
        DATA_DIR,
        "y_train_mc.npy"
    )
)

X_test = np.load(
    os.path.join(
        DATA_DIR,
        "X_test_mc.npy"
    )
)

y_test = np.load(
    os.path.join(
        DATA_DIR,
        "y_test_mc.npy"
    )
)


input_dim = X_train.shape[1]

n = len(X_train)
split = n // 3


nodes = [

    MultiClassNode(
        1,
        X_train[:split],
        y_train[:split]
    ),

    MultiClassNode(
        2,
        X_train[split:2 * split],
        y_train[split:2 * split]
    ),

    MultiClassNode(
        3,
        X_train[2 * split:],
        y_train[2 * split:]
    )
]


sample_counts = [
    len(node.X_t)
    for node in nodes
]


global_model = MultiClassIDS(
    input_dim=input_dim
)


X_test_t = torch.as_tensor(
    X_test,
    dtype=torch.float32
)


history = []

ROUNDS = 15
LOCAL_EPOCHS = 3


print(
    "\n===== FEDERATED MULTI-CLASS TRAINING ====="
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
            epochs=LOCAL_EPOCHS
        )

        losses.append(loss)

    averaged_weights = fed_avg(
        [
            node.get_weights()
            for node in nodes
        ],
        sample_counts
    )

    global_model.set_weights(
        averaged_weights
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
        f"{macro_f1:.4f} | "
        f"Loss: {avg_loss:.4f}"
    )


print(
    "\n===== FINAL MULTI-CLASS REPORT ====="
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


model_path = os.path.join(
    MODELS_DIR,
    "federated_multiclass_model.pth"
)

history_path = os.path.join(
    MODELS_DIR,
    "federated_multiclass_history.json"
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
    f"Model saved to {model_path}"
)