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

from model import IntrusionDetector


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


loader = DataLoader(
    TensorDataset(
        X_train_t,
        y_train_t
    ),
    batch_size=256,
    shuffle=True
)


model = IntrusionDetector(
    input_dim=input_dim
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.BCEWithLogitsLoss()


history = []

EPOCHS = 20


print(
    "Training centralized baseline..."
)


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    total_loss = 0.0
    batches = 0

    for X_batch, y_batch in loader:

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
        ).long().view(-1)

    f1 = f1_score(
        y_test,
        predictions.numpy(),
        zero_division=0
    )

    history.append(
        {
            "epoch": epoch,
            "loss": avg_loss,
            "f1": float(f1)
        }
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {avg_loss:.4f} | "
        f"F1: {f1:.4f}"
    )


print(
    "\n===== FINAL CLASSIFICATION REPORT ====="
)


print(
    classification_report(
        y_test,
        predictions.numpy(),
        target_names=[
            "Normal",
            "Attack"
        ],
        zero_division=0
    )
)


model_path = os.path.join(
    MODELS_DIR,
    "baseline_model.pth"
)

history_path = os.path.join(
    MODELS_DIR,
    "baseline_history.json"
)


torch.save(
    model.state_dict(),
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
    f"Baseline model saved to {model_path}"
)