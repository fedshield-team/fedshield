import json
import os

import numpy as np
import torch
import torch.nn as nn

from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

from model import MultiClassIDS


CLASS_NAMES = [
    "Normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R"
]


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

os.makedirs(MODELS_DIR, exist_ok=True)


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


print(
    "Original class distribution:"
)

for i, name in enumerate(
    CLASS_NAMES
):

    print(
        f"  {name}: "
        f"{np.sum(y_train == i)}"
    )


print(
    "\nApplying SMOTE..."
)

smote = SMOTE(
    random_state=42,
    k_neighbors=5
)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)


print(
    "\nClass distribution after SMOTE:"
)

for i, name in enumerate(
    CLASS_NAMES
):

    print(
        f"  {name}: "
        f"{np.sum(y_train == i)}"
    )


input_dim = X_train.shape[1]

X_train_t = torch.as_tensor(
    X_train,
    dtype=torch.float32
)

y_train_t = torch.as_tensor(
    y_train,
    dtype=torch.long
)

X_test_t = torch.as_tensor(
    X_test,
    dtype=torch.float32
)

y_test_t = torch.as_tensor(
    y_test,
    dtype=torch.long
)


loader = DataLoader(
    TensorDataset(
        X_train_t,
        y_train_t
    ),
    batch_size=256,
    shuffle=True
)


model = MultiClassIDS(
    input_dim=input_dim,
    num_classes=len(CLASS_NAMES)
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=3,
    factor=0.5
)


criterion = nn.CrossEntropyLoss()

history = []

EPOCHS = 30


print(
    "\n============================================================"
)

print(
    "CENTRALIZED MULTI-CLASS TRAINING"
)

print(
    "============================================================"
)


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    total_loss = 0.0
    batches = 0

    for X_batch, y_batch in loader:

        optimizer.zero_grad(
            set_to_none=True
        )

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

    scheduler.step(
        avg_loss
    )

    model.eval()

    with torch.no_grad():

        predictions = (
            model(
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

    history.append(
        {
            "epoch": epoch,
            "loss": float(avg_loss),
            "macro_f1": float(macro_f1)
        }
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {avg_loss:.4f} | "
        f"Macro F1: {macro_f1:.4f}"
    )


print(
    "\n============================================================"
)

print(
    "FINAL CLASSIFICATION REPORT"
)

print(
    "============================================================"
)

print(
    classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


model_path = os.path.join(
    MODELS_DIR,
    "multiclass_model.pth"
)

history_path = os.path.join(
    MODELS_DIR,
    "multiclass_history.json"
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
    f"Multi-class model saved to:\n{model_path}"
)

print(
    f"History saved to:\n{history_path}"
)