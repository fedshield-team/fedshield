import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from model import IntrusionDetector


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


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


MODEL_OUT = os.path.join(
    MODELS_DIR,
    "cicids2017_model.pth"
)

HISTORY_OUT = os.path.join(
    MODELS_DIR,
    "cicids2017_history.json"
)


EPOCHS = 20
BATCH_SIZE = 256
LR = 0.001


def load_data():

    print(
        "Loading CICIDS2017 preprocessed arrays..."
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

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_test : {X_test.shape}"
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test
    )


def train():

    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = load_data()

    input_dim = X_train.shape[1]

    print(
        f"Detected input features: "
        f"{input_dim}"
    )

    X_train_t = torch.as_tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_t = torch.as_tensor(
        y_train,
        dtype=torch.float32
    ).view(-1, 1)

    X_test_t = torch.as_tensor(
        X_test,
        dtype=torch.float32
    )

    train_loader = DataLoader(
        TensorDataset(
            X_train_t,
            y_train_t
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = IntrusionDetector(
        input_dim=input_dim
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    history = []

    best_f1 = -1.0

    best_predictions = None


    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0
        batches = 0

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = (
            total_loss /
            max(batches, 1)
        )

        model.eval()

        with torch.no_grad():

            probabilities = torch.sigmoid(
                model(X_test_t)
            )

            predictions = (
                probabilities >= 0.5
            ).long().view(-1).numpy()

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall)
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"F1: {f1:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f}"
        )

        if f1 > best_f1:

            best_f1 = f1

            best_predictions = predictions.copy()

            torch.save(
                model.state_dict(),
                MODEL_OUT
            )

            print(
                f"  ✓ Best model saved "
                f"(F1: {best_f1:.4f})"
            )


    with open(
        HISTORY_OUT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )


    print(
        "\n===== CICIDS2017 RESULTS ====="
    )

    cm = confusion_matrix(
        y_test,
        best_predictions
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)

    print(
        f"\nBest F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"Model: {MODEL_OUT}"
    )

    print(
        f"History: {HISTORY_OUT}"
    )

    return best_f1


if __name__ == "__main__":

    train()