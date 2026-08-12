"""
FedShield — Online Incremental Retraining

Design constraints this respects:
  1. Labels must come from an independent, rule-based signal — NOT the model's own
     predictions. Using self-predictions as training labels creates a feedback loop
     that reinforces whatever the model already believes (including its mistakes).
     The only label source here is live_capture.py's check_port_scan(), which flags
     Probe/scan traffic using a fixed heuristic (N distinct ports in a time window),
     completely independent of the neural net's output.
  2. Retraining is a SHORT fine-tune of the existing global model, not a full
     from-scratch federated run — this is what makes it "online"/incremental
     instead of a 15-round batch job.
  3. A small replay sample from the original training set is mixed in on every
     fine-tune to reduce catastrophic forgetting of the other 4 classes.
  4. Before accepting new weights, macro-F1 on the held-out test set is compared
     before vs. after. If it regresses, the new weights are discarded and the old
     model is kept — an online update is only as good as its safety guard.
  5. No process restart needed: accepted updates bump a version file that
     live_capture.py polls and hot-reloads.

Usage (called from live_capture.py, or standalone for testing):
    from online_retrain import log_retrain_sample, should_retrain, run_incremental_retrain
"""

import os
import json
import sqlite3
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH        = os.path.join(BASE, "models", "fedshield_logs.db")
MODEL_PATH     = os.path.join(BASE, "models", "federated_noniid_model.pth")
MODEL_BACKUP   = os.path.join(BASE, "models", "federated_noniid_model_prev.pth")
HISTORY_PATH   = os.path.join(BASE, "models", "federated_noniid_history.json")
VERSION_PATH   = os.path.join(BASE, "models", "model_version.json")
DRIFT_LOG      = os.path.join(BASE, "models", "drift_log.json")

X_TEST_PATH    = os.path.join(BASE, "data", "X_test_mc.npy")
Y_TEST_PATH    = os.path.join(BASE, "data", "y_test_mc.npy")
X_TRAIN_PATH   = os.path.join(BASE, "data", "X_train_mc.npy")
Y_TRAIN_PATH   = os.path.join(BASE, "data", "y_train_mc.npy")

CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

# ── Trigger config ────────────────────────────────────────────────────────────
MIN_BUFFER_SAMPLES   = 40      # retrain once this many rule-confirmed samples accumulate
REPLAY_SAMPLE_SIZE   = 300     # samples drawn from original train set to prevent forgetting
FINE_TUNE_EPOCHS     = 2
FINE_TUNE_LR         = 0.0005
F1_REGRESSION_TOLERANCE = 0.0  # new F1 must be >= old F1 to accept (strict, no regression allowed)


class MultiClassIDS(nn.Module):
    """Must match the architecture in live_capture.py / federated_noniid.py exactly —
    fine-tuning loads into this shape."""
    def __init__(self, input_dim=41, num_classes=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.network(x)


# ── Buffer (rule-confirmed samples only) ──────────────────────────────────────
def init_retrain_buffer():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS retrain_buffer (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT,
            features   TEXT,   -- JSON list of 41 floats
            label      INTEGER,
            source     TEXT,   -- e.g. 'rule_confirmed_probe'
            consumed   INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def log_retrain_sample(features: np.ndarray, label: int, source: str):
    """Call this ONLY with labels from an independent, non-model signal."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO retrain_buffer (timestamp, features, label, source) VALUES (?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), json.dumps(features.tolist()), int(label), source)
    )
    conn.commit()
    conn.close()


def _unconsumed_buffer_count() -> int:
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM retrain_buffer WHERE consumed=0").fetchone()[0]
    conn.close()
    return count


def _load_unconsumed_buffer():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, features, label FROM retrain_buffer WHERE consumed=0").fetchall()
    conn.close()
    ids = [r[0] for r in rows]
    X = np.array([json.loads(r[1]) for r in rows], dtype=np.float32)
    y = np.array([r[2] for r in rows], dtype=np.int64)
    return ids, X, y


def _mark_consumed(ids):
    if not ids:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.executemany("UPDATE retrain_buffer SET consumed=1 WHERE id=?", [(i,) for i in ids])
    conn.commit()
    conn.close()


# ── Trigger check ──────────────────────────────────────────────────────────────
def _latest_drift_is_critical() -> bool:
    try:
        with open(DRIFT_LOG) as f:
            logs = json.load(f)
        return bool(logs) and logs[-1].get("status") == "DRIFT_DETECTED"
    except Exception:
        return False


def should_retrain() -> bool:
    """True if either (a) enough rule-confirmed samples have accumulated, or
    (b) the last drift_detection.py run flagged CRITICAL drift."""
    return _unconsumed_buffer_count() >= MIN_BUFFER_SAMPLES or _latest_drift_is_critical()


# ── Retraining ─────────────────────────────────────────────────────────────────
def _evaluate_f1(model, X_test, y_test) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).argmax(dim=1).numpy()
    return f1_score(y_test, preds, average="macro")


def run_incremental_retrain(verbose=True) -> dict:
    """
    Runs a short fine-tune on the buffered rule-confirmed samples + a replay sample,
    evaluates against the held-out test set, and only commits the update if F1 doesn't
    regress. Returns a summary dict either way — caller decides what to do with it.
    """
    result = {"triggered_at": datetime.utcnow().isoformat(), "accepted": False}

    if not (os.path.exists(X_TEST_PATH) and os.path.exists(Y_TEST_PATH)):
        result["error"] = "Missing data/X_test_mc.npy or y_test_mc.npy — cannot safety-check retrain."
        return result

    ids, X_new, y_new = _load_unconsumed_buffer()
    if len(X_new) == 0 and not _latest_drift_is_critical():
        result["error"] = "No buffered samples and no critical drift — nothing to retrain on."
        return result

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    # Replay sample from original training data, stratified-ish via random draw
    X_replay = y_replay = None
    if os.path.exists(X_TRAIN_PATH) and os.path.exists(Y_TRAIN_PATH):
        X_train_full = np.load(X_TRAIN_PATH)
        y_train_full = np.load(Y_TRAIN_PATH)
        n = min(REPLAY_SAMPLE_SIZE, len(X_train_full))
        idx = np.random.choice(len(X_train_full), n, replace=False)
        X_replay, y_replay = X_train_full[idx], y_train_full[idx]

    model = MultiClassIDS()
    if not os.path.exists(MODEL_PATH):
        result["error"] = f"No existing model at {MODEL_PATH} to fine-tune."
        return result
    model.load_state_dict(torch.load(MODEL_PATH))

    f1_before = _evaluate_f1(model, X_test, y_test)

    # Build fine-tune set: new buffered samples + replay
    if X_replay is not None and len(X_new) > 0:
        X_ft = np.concatenate([X_new, X_replay])
        y_ft = np.concatenate([y_new, y_replay])
    elif X_replay is not None:
        X_ft, y_ft = X_replay, y_replay
    else:
        X_ft, y_ft = X_new, y_new

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_ft), torch.LongTensor(y_ft)),
        batch_size=64, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(FINE_TUNE_EPOCHS):
        for X_b, y_b in loader:
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()

    f1_after = _evaluate_f1(model, X_test, y_test)

    result.update({
        "buffer_samples_used": len(X_new),
        "replay_samples_used": 0 if X_replay is None else len(X_replay),
        "f1_before": round(float(f1_before), 4),
        "f1_after":  round(float(f1_after), 4),
    })

    if f1_after + F1_REGRESSION_TOLERANCE < f1_before:
        result["accepted"] = False
        result["reason"] = "F1 regressed — new weights discarded, old model kept."
        _mark_consumed(ids)  # don't retry on the same stale samples forever
        if verbose:
            print(f"[online_retrain] REJECTED: F1 {f1_before:.4f} -> {f1_after:.4f} (regression)")
        return result

    # Accept: backup old model, save new, bump version, log a history round
    if os.path.exists(MODEL_PATH):
        torch.save(torch.load(MODEL_PATH), MODEL_BACKUP)
    torch.save(model.state_dict(), MODEL_PATH)

    try:
        with open(HISTORY_PATH) as f:
            history = json.load(f)
    except Exception:
        history = []
    next_round = (history[-1]["round"] + 1) if history else 1
    history.append({
        "round": next_round,
        "macro_f1": round(float(f1_after), 4),
        "online_retrain": True,
        "buffer_samples": len(X_new),
        "timestamp": result["triggered_at"],
    })
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)

    version = int(time.time())
    with open(VERSION_PATH, "w") as f:
        json.dump({"version": version, "updated_at": result["triggered_at"], "round": next_round}, f)

    _mark_consumed(ids)
    result["accepted"] = True
    result["reason"] = "F1 held or improved — new weights committed."
    result["version"] = version

    if verbose:
        print(f"[online_retrain] ACCEPTED: F1 {f1_before:.4f} -> {f1_after:.4f} "
              f"({len(X_new)} new samples, {result['replay_samples_used']} replay). "
              f"Model version {version} written.")

    return result


if __name__ == "__main__":
    init_retrain_buffer()
    print(f"Buffered unconsumed samples: {_unconsumed_buffer_count()}")
    if should_retrain():
        print(run_incremental_retrain())
    else:
        print(f"Not enough samples yet (need {MIN_BUFFER_SAMPLES}) and no critical drift logged.")