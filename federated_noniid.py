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

from model import MultiClassIDS
from server.aggregator import fed_avg


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = [
    "Normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R"
]

NUM_CLASSES = len(
    CLASS_NAMES
)


# ============================================================
# PATHS
# ============================================================

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


# ============================================================
# ARGUMENTS
# ============================================================

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


# ============================================================
# DATASET PATHS
# ============================================================

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


# ============================================================
# LOAD DATA
# ============================================================

print(
    f"\n===== DATASET: {args.dataset} ====="
)

X_train = np.load(
    X_TRAIN_PATH
)

y_train = np.load(
    Y_TRAIN_PATH
)

X_test = np.load(
    X_TEST_PATH
)

y_test = np.load(
    Y_TEST_PATH
)

input_dim = X_train.shape[1]

print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Features: {input_dim}"
)

if input_dim != 41:

    raise ValueError(
        f"FedShield expects 41 features, "
        f"got {input_dim}"
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print(
    "\n===== GLOBAL TRAINING DISTRIBUTION ====="
)

for class_id, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{class_name}: "
        f"{int(np.sum(y_train == class_id))}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    y_train,
    minlength=NUM_CLASSES
).astype(
    np.float32
)

class_weights = (
    len(y_train)
    /
    (
        NUM_CLASSES
        *
        np.maximum(
            class_counts,
            1.0
        )
    )
)

# Keep the weights numerically reasonable.
class_weights = np.clip(
    class_weights,
    1.0,
    10.0
)

print(
    "\n===== CLASS WEIGHTS ====="
)

for i, name in enumerate(
    CLASS_NAMES
):

    print(
        f"{name}: "
        f"{class_weights[i]:.4f}"
    )


# ============================================================
# TRUE NON-IID PARTITION
# ============================================================

def create_noniid_partition(
    y,
    seed=42
):

    rng = np.random.default_rng(
        seed
    )

    client_indices = [
        [],
        [],
        []
    ]

    # Desired approximate client proportions
    # for each class.
    #
    # Hospital: Normal-heavy
    # Bank: DoS/Probe-heavy
    # Campus: receives remaining data

    proportions = {
        0: [0.50, 0.25, 0.25],  # Normal
        1: [0.10, 0.60, 0.30],  # DoS
        2: [0.10, 0.60, 0.30],  # Probe
        3: [0.60, 0.20, 0.20],  # R2L
        4: [0.30, 0.30, 0.40],  # U2R
    }

    for class_id in range(
        NUM_CLASSES
    ):

        indices = np.where(
            y == class_id
        )[0]

        indices = indices.copy()

        rng.shuffle(
            indices
        )

        p = proportions[
            class_id
        ]

        n = len(indices)

        n1 = int(
            n * p[0]
        )

        n2 = int(
            n * p[1]
        )

        first_end = n1

        second_end = (
            n1 + n2
        )

        client_indices[0].extend(
            indices[
                :first_end
            ].tolist()
        )

        client_indices[1].extend(
            indices[
                first_end:second_end
            ].tolist()
        )

        client_indices[2].extend(
            indices[
                second_end:
            ].tolist()
        )

    result = []

    for indices in client_indices:

        indices = np.asarray(
            indices,
            dtype=np.int64
        )

        rng.shuffle(
            indices
        )

        result.append(
            indices
        )

    return result


node_indices = create_noniid_partition(
    y_train
)


# ============================================================
# VERIFY PARTITION
# ============================================================

all_assigned = np.concatenate(
    node_indices
)

if len(all_assigned) != len(
    y_train
):

    raise RuntimeError(
        "Non-IID partition does not "
        "contain every training sample."
    )

if len(
    np.unique(all_assigned)
) != len(
    y_train
):

    raise RuntimeError(
        "Non-IID partition contains "
        "duplicate samples."
    )


# ============================================================
# NODE
# ============================================================

class MultiClassNode:

    def __init__(
        self,
        node_id,
        X,
        y,
        label,
        lr,
        loss_weights
    ):

        self.node_id = node_id
        self.label = label
        self.lr = lr

        self.model = MultiClassIDS(
            input_dim=X.shape[1],
            num_classes=NUM_CLASSES
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        self.X_t = torch.as_tensor(
            X,
            dtype=torch.float32
        )

        self.y_t = torch.as_tensor(
            y,
            dtype=torch.long
        )

        self.loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                self.X_t,
                self.y_t
            ),
            batch_size=256,
            shuffle=True
        )

        self.criterion = nn.CrossEntropyLoss(
            weight=loss_weights
        )

        unique, counts = np.unique(
            y,
            return_counts=True
        )

        distribution = {
            CLASS_NAMES[int(label)]:
            int(count)
            for label, count
            in zip(
                unique,
                counts
            )
        }

        print(
            f"[Node {node_id} - {label}] "
            f"{len(X)} samples | "
            f"Distribution: {distribution}"
        )

    def train_local(
        self,
        epochs=3
    ):

        self.model.train()

        total_loss = 0.0
        batches = 0

        for _ in range(
            epochs
        ):

            for X_batch, y_batch in self.loader:

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                logits = self.model(
                    X_batch
                )

                loss = self.criterion(
                    logits,
                    y_batch
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=5.0
                )

                self.optimizer.step()

                total_loss += loss.item()
                batches += 1

        return (
            total_loss /
            max(
                batches,
                1
            )
        )

    def get_weights(self):

        return self.model.get_weights()

    def set_weights(
        self,
        weights
    ):

        self.model.set_weights(
            weights
        )

        # Reset local optimizer after
        # receiving the global model.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

if args.dataset == "cicids2017":

    lr = 0.0003
    local_epochs = 1

else:

    lr = 0.001
    local_epochs = 3


loss_weights_tensor = torch.as_tensor(
    class_weights,
    dtype=torch.float32
)


# ============================================================
# CREATE NODES
# ============================================================

print(
    "\n===== NON-IID NODE DISTRIBUTION ====="
)

nodes = [
    MultiClassNode(
        1,
        X_train[node_indices[0]],
        y_train[node_indices[0]],
        "Hospital",
        lr,
        loss_weights_tensor
    ),

    MultiClassNode(
        2,
        X_train[node_indices[1]],
        y_train[node_indices[1]],
        "Bank",
        lr,
        loss_weights_tensor
    ),

    MultiClassNode(
        3,
        X_train[node_indices[2]],
        y_train[node_indices[2]],
        "Campus",
        lr,
        loss_weights_tensor
    )
]


sample_counts = [
    len(node.X_t)
    for node in nodes
]


print(
    "\nTotal assigned samples: "
    f"{sum(sample_counts)}"
)


# ============================================================
# GLOBAL MODEL
# ============================================================

global_model = MultiClassIDS(
    input_dim=input_dim,
    num_classes=NUM_CLASSES
)

X_test_t = torch.as_tensor(
    X_test,
    dtype=torch.float32
)


# ============================================================
# FEDERATED TRAINING
# ============================================================

ROUNDS = 15

history = []

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

    # --------------------------------------------------------
    # Send global model to every node
    # --------------------------------------------------------

    for node in nodes:

        node.set_weights(
            global_weights
        )

    # --------------------------------------------------------
    # Local training
    # --------------------------------------------------------

    losses = []

    for node in nodes:

        loss = node.train_local(
            epochs=local_epochs
        )

        losses.append(
            loss
        )

    # --------------------------------------------------------
    # FedAvg
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Global evaluation
    # --------------------------------------------------------

    global_model.eval()

    with torch.no_grad():

        predictions = (
            global_model(
                X_test_t
            )
            .argmax(
                dim=1
            )
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
            "macro_f1": float(
                macro_f1
            )
        }
    )

    print(
        f"Global Macro F1: "
        f"{macro_f1:.4f} | "
        f"Loss: {avg_loss:.4f}"
    )


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "\n===== FINAL REPORT ====="
)

print(
    classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

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
    f"Model saved to:\n{OUT_MODEL}"
)

print(
    f"History saved to:\n{OUT_HISTORY}"
)