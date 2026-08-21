"""FedShield — safe online incremental retraining."""

import json
import os
import sqlite3
import tempfile

from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import f1_score

from torch.utils.data import (
    DataLoader,
    TensorDataset
)


BASE = os.path.dirname(
    os.path.abspath(__file__)
)

DB_PATH = os.path.join(
    BASE,
    "models",
    "fedshield_logs.db"
)

MODEL_PATH = os.path.join(
    BASE,
    "models",
    "federated_noniid_model.pth"
)

MODEL_BACKUP = os.path.join(
    BASE,
    "models",
    "federated_noniid_model_prev.pth"
)

HISTORY_PATH = os.path.join(
    BASE,
    "models",
    "federated_noniid_history.json"
)

VERSION_PATH = os.path.join(
    BASE,
    "models",
    "model_version.json"
)

DRIFT_LOG = os.path.join(
    BASE,
    "models",
    "drift_log.json"
)

X_TEST_PATH = os.path.join(
    BASE,
    "data",
    "X_test_mc.npy"
)

Y_TEST_PATH = os.path.join(
    BASE,
    "data",
    "y_test_mc.npy"
)

X_TRAIN_PATH = os.path.join(
    BASE,
    "data",
    "X_train_mc.npy"
)

Y_TRAIN_PATH = os.path.join(
    BASE,
    "data",
    "y_train_mc.npy"
)

CLASS_NAMES = [
    "Normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R"
]


MIN_BUFFER_SAMPLES = 40

REPLAY_SAMPLE_SIZE = 300

FINE_TUNE_EPOCHS = 2

FINE_TUNE_LR = 0.0005

F1_REGRESSION_TOLERANCE = 0.0


class MultiClassIDS(
    nn.Module
):

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

            nn.BatchNorm1d(
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                256,
                128
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            ),

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


def _connect():

    os.makedirs(
        os.path.dirname(
            DB_PATH
        ),
        exist_ok=True
    )

    return sqlite3.connect(
        DB_PATH,
        timeout=10
    )


def init_retrain_buffer():

    with _connect() as conn:

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS retrain_buffer (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                features TEXT NOT NULL,
                label INTEGER NOT NULL,
                source TEXT NOT NULL,
                consumed INTEGER DEFAULT 0
            )
            """
        )


def log_retrain_sample(
    features,
    label,
    source
):

    features = np.asarray(
        features,
        dtype=np.float32
    ).reshape(-1)

    if features.size != 41:

        raise ValueError(
            f"Expected 41 features, "
            f"got {features.size}"
        )

    if not 0 <= int(label) < len(
        CLASS_NAMES
    ):

        raise ValueError(
            "Invalid class label"
        )

    with _connect() as conn:

        conn.execute(
            """
            INSERT INTO retrain_buffer
            (
                timestamp,
                features,
                label,
                source
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                datetime.now(
                    timezone.utc
                ).isoformat(),

                json.dumps(
                    features.tolist()
                ),

                int(label),

                source
            )
        )


def _unconsumed_buffer_count():

    with _connect() as conn:

        return conn.execute(
            """
            SELECT COUNT(*)
            FROM retrain_buffer
            WHERE consumed=0
            """
        ).fetchone()[0]


def _load_unconsumed_buffer():

    with _connect() as conn:

        rows = conn.execute(
            """
            SELECT id, features, label
            FROM retrain_buffer
            WHERE consumed=0
            ORDER BY id
            """
        ).fetchall()

    ids = [
        r[0]
        for r in rows
    ]

    if not rows:

        return (
            ids,
            np.empty(
                (0, 41),
                np.float32
            ),
            np.empty(
                (0,),
                np.int64
            )
        )

    X = np.asarray(
        [
            json.loads(r[1])
            for r in rows
        ],
        dtype=np.float32
    )

    y = np.asarray(
        [
            r[2]
            for r in rows
        ],
        dtype=np.int64
    )

    return ids, X, y


def _mark_consumed(
    ids
):

    if not ids:
        return

    with _connect() as conn:

        conn.executemany(
            """
            UPDATE retrain_buffer
            SET consumed=1
            WHERE id=?
            """,
            [
                (i,)
                for i in ids
            ]
        )


def _latest_drift_is_critical():

    try:

        with open(
            DRIFT_LOG,
            encoding="utf-8"
        ) as f:

            logs = json.load(f)

        return (
            isinstance(
                logs,
                list
            )
            and bool(logs)
            and
            logs[-1].get(
                "status"
            )
            == "DRIFT_DETECTED"
        )

    except (
        OSError,
        ValueError,
        json.JSONDecodeError
    ):

        return False


def should_retrain():

    return (
        _unconsumed_buffer_count()
        >= MIN_BUFFER_SAMPLES
        or
        _latest_drift_is_critical()
    )


def _evaluate_f1(
    model,
    X,
    y
):

    model.eval()

    with torch.no_grad():

        pred = (
            model(
                torch.as_tensor(
                    X,
                    dtype=torch.float32
                )
            )
            .argmax(1)
            .cpu()
            .numpy()
        )

    return float(
        f1_score(
            y,
            pred,
            average="macro",
            zero_division=0
        )
    )


def _atomic_torch_save(
    state,
    path
):

    directory = os.path.dirname(
        path
    )

    fd, tmp = tempfile.mkstemp(
        prefix=".model_",
        suffix=".pth",
        dir=directory
    )

    os.close(fd)

    try:

        torch.save(
            state,
            tmp
        )

        os.replace(
            tmp,
            path
        )

    finally:

        if os.path.exists(tmp):

            os.remove(tmp)


def _atomic_json_save(
    data,
    path
):

    directory = os.path.dirname(
        path
    )

    fd, tmp = tempfile.mkstemp(
        prefix=".json_",
        suffix=".tmp",
        dir=directory,
        text=True
    )

    try:

        with os.fdopen(
            fd,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=2
            )

        os.replace(
            tmp,
            path
        )

    finally:

        if os.path.exists(tmp):

            os.remove(tmp)


def run_incremental_retrain(
    verbose=True
):

    result = {
        "triggered_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "accepted":
            False
    }

    init_retrain_buffer()

    if not (
        os.path.exists(X_TEST_PATH)
        and
        os.path.exists(Y_TEST_PATH)
    ):

        result["error"] = (
            "Missing held-out test data."
        )

        return result

    ids, X_new, y_new = (
        _load_unconsumed_buffer()
    )

    critical = (
        _latest_drift_is_critical()
    )

    if (
        len(X_new) == 0
        and not critical
    ):

        result["error"] = (
            "No buffered samples "
            "and no critical drift."
        )

        return result

    if not os.path.exists(
        MODEL_PATH
    ):

        result["error"] = (
            f"No model at {MODEL_PATH}."
        )

        return result

    X_test = np.load(
        X_TEST_PATH
    )

    y_test = np.load(
        Y_TEST_PATH
    )

    if (
        X_test.ndim != 2
        or
        X_test.shape[1] != 41
    ):

        result["error"] = (
            "Invalid test feature shape."
        )

        return result

    X_replay = None
    y_replay = None

    if (
        os.path.exists(X_TRAIN_PATH)
        and
        os.path.exists(Y_TRAIN_PATH)
    ):

        X_full = np.load(
            X_TRAIN_PATH
        )

        y_full = np.load(
            Y_TRAIN_PATH
        )

        if len(X_full):

            n = min(
                REPLAY_SAMPLE_SIZE,
                len(X_full)
            )

            idx = np.random.choice(
                len(X_full),
                n,
                replace=False
            )

            X_replay = X_full[
                idx
            ]

            y_replay = y_full[
                idx
            ]

    model = MultiClassIDS()

    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        state
    )

    f1_before = _evaluate_f1(
        model,
        X_test,
        y_test
    )

    if (
        X_replay is not None
        and
        len(X_new)
    ):

        X_ft = np.concatenate(
            [
                X_new,
                X_replay
            ]
        )

        y_ft = np.concatenate(
            [
                y_new,
                y_replay
            ]
        )

    elif X_replay is not None:

        X_ft = X_replay
        y_ft = y_replay

    else:

        X_ft = X_new
        y_ft = y_new

    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(
                X_ft,
                dtype=torch.float32
            ),
            torch.as_tensor(
                y_ft,
                dtype=torch.long
            )
        ),
        batch_size=64,
        shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FINE_TUNE_LR
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    model.train()

    for _ in range(
        FINE_TUNE_EPOCHS
    ):

        for xb, yb in loader:

            optimizer.zero_grad(
                set_to_none=True
            )

            loss = criterion(
                model(xb),
                yb
            )

            loss.backward()

            optimizer.step()

    f1_after = _evaluate_f1(
        model,
        X_test,
        y_test
    )

    result.update(
        {
            "buffer_samples_used":
                len(X_new),

            "replay_samples_used":
                (
                    0
                    if X_replay is None
                    else len(X_replay)
                ),

            "f1_before":
                round(
                    f1_before,
                    4
                ),

            "f1_after":
                round(
                    f1_after,
                    4
                )
        }
    )

    if (
        f1_after
        + F1_REGRESSION_TOLERANCE
        < f1_before
    ):

        result["reason"] = (
            "F1 regressed — "
            "new weights discarded."
        )

        _mark_consumed(
            ids
        )

        return result

    # Backup old model
    if os.path.exists(
        MODEL_PATH
    ):

        old_state = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )

        _atomic_torch_save(
            old_state,
            MODEL_BACKUP
        )

    # Save new model atomically
    _atomic_torch_save(
        model.state_dict(),
        MODEL_PATH
    )

    try:

        with open(
            HISTORY_PATH,
            encoding="utf-8"
        ) as f:

            history = json.load(f)

        if not isinstance(
            history,
            list
        ):

            history = []

    except (
        OSError,
        ValueError,
        json.JSONDecodeError
    ):

        history = []

    next_round = (
        int(
            history[-1].get(
                "round",
                0
            )
        )
        + 1
        if history
        else 1
    )

    history.append(
        {
            "round":
                next_round,

            "macro_f1":
                round(
                    f1_after,
                    4
                ),

            "online_retrain":
                True,

            "buffer_samples":
                len(X_new),

            "timestamp":
                result[
                    "triggered_at"
                ]
        }
    )

    _atomic_json_save(
        history,
        HISTORY_PATH
    )

    version = int(
        datetime.now(
            timezone.utc
        ).timestamp()
        * 1000
    )

    _atomic_json_save(
        {
            "version": version,
            "updated_at":
                result[
                    "triggered_at"
                ],
            "round":
                next_round
        },
        VERSION_PATH
    )

    _mark_consumed(
        ids
    )

    result.update(
        {
            "accepted": True,

            "reason":
                "F1 held or improved — "
                "new weights committed.",

            "version":
                version
        }
    )

    if verbose:

        print(
            f"[online_retrain] "
            f"ACCEPTED: F1 "
            f"{f1_before:.4f} -> "
            f"{f1_after:.4f}; "
            f"version={version}"
        )

    return result


if __name__ == "__main__":

    init_retrain_buffer()

    print(
        f"Buffered samples: "
        f"{_unconsumed_buffer_count()}"
    )

    if should_retrain():

        print(
            run_incremental_retrain()
        )

    else:

        print(
            f"Not enough samples yet "
            f"(need {MIN_BUFFER_SAMPLES}) "
            f"and no critical drift."
        )