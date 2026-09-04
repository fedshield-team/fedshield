"""Flower server for the authoritative FedShield multiclass model path."""

from __future__ import annotations

import json
import os
import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.metrics import classification_report

from model import (
    MULTICLASS_CLASS_NAMES,
    MULTICLASS_INPUT_DIM,
    MULTICLASS_NUM_CLASSES,
    MultiClassIDS,
)


BASE = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE / "models"
MODEL_PATH = MODELS_DIR / "federated_noniid_model.pth"
VERSION_PATH = MODELS_DIR / "model_version.json"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

round_metrics = []


def load_authoritative_model(model_path: Path = MODEL_PATH) -> MultiClassIDS:
    """Load the active multiclass artifact used by fedshield_runtime."""
    model = MultiClassIDS(
        input_dim=MULTICLASS_INPUT_DIM,
        num_classes=MULTICLASS_NUM_CLASSES,
    )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Authoritative multiclass model artifact not found: {model_path}"
        )
    state = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _model_parameters(model: MultiClassIDS):
    return [
        parameter.detach().cpu().numpy()
        for parameter in model.parameters()
    ]


def save_authoritative_model(
    model: MultiClassIDS,
    parameters,
    model_path: Path = MODEL_PATH,
) -> None:
    """Validate and persist Flower's aggregated state as the active artifact."""
    weights = [
        torch.as_tensor(array, dtype=parameter.dtype)
        for parameter, array in zip(model.parameters(), parameters)
    ]
    model.set_weights(weights)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".model_",
        suffix=".pth",
        dir=model_path.parent,
    )
    os.close(fd)
    try:
        torch.save(model.state_dict(), temporary_path)
        with open(temporary_path, "rb") as file:
            model_sha256 = hashlib.sha256(file.read()).hexdigest()
        os.replace(temporary_path, model_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise

    version = int(datetime.now(timezone.utc).timestamp() * 1_000_000)
    with VERSION_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "version": version,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "sha256": model_sha256,
            },
            file,
        )


def evaluate_final_model(model: MultiClassIDS) -> None:
    """Evaluate the final saved global model and atomically publish its report."""
    data_dir = BASE / "data"
    evaluation_path = MODELS_DIR / "federated_noniid_evaluation.json"
    test_features_path = data_dir / "X_test_mc.npy"
    test_labels_path = data_dir / "y_test_mc.npy"

    X_test = np.load(test_features_path)
    y_test = np.load(test_labels_path)

    if X_test.ndim != 2 or X_test.shape[1] != MULTICLASS_INPUT_DIM:
        raise ValueError(
            f"Expected test data with {MULTICLASS_INPUT_DIM} features; "
            f"got {X_test.shape}"
        )

    if len(X_test) != len(y_test):
        raise ValueError("Test features and labels must have matching lengths")

    model.eval()
    with torch.no_grad():
        predictions = (
            model(torch.as_tensor(X_test, dtype=torch.float32))
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

    report = classification_report(
        y_test,
        predictions,
        target_names=MULTICLASS_CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    with MODEL_PATH.open("rb") as file:
        model_sha256 = hashlib.sha256(file.read()).hexdigest()

    with VERSION_PATH.open(encoding="utf-8") as file:
        model_version = json.load(file)["version"]

    evaluation = {
        "model": "federated_noniid_model",
        "model_version": model_version,
        "model_sha256": model_sha256,
        "class_order": MULTICLASS_CLASS_NAMES,
        "test_data": {
            "path": os.path.relpath(test_features_path, BASE),
            "shape": list(X_test.shape),
            "labels_path": os.path.relpath(test_labels_path, BASE),
            "labels_shape": list(y_test.shape),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in MULTICLASS_CLASS_NAMES
        },
        "aggregate": {
            "accuracy": float(report["accuracy"]),
            "macro_avg": report["macro avg"],
            "weighted_avg": report["weighted avg"],
        },
    }

    fd, temporary_path = tempfile.mkstemp(
        prefix=".evaluation_",
        suffix=".json",
        dir=evaluation_path.parent,
    )
    os.close(fd)
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(evaluation, file, indent=2)
        os.replace(temporary_path, evaluation_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


class MulticlassFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy that persists every valid global multiclass state."""

    def __init__(self, global_model: MultiClassIDS, model_path: Path = MODEL_PATH, **kwargs):
        self.global_model = global_model
        self.model_path = model_path
        super().__init__(
            initial_parameters=ndarrays_to_parameters(
                _model_parameters(global_model)
            ),
            **kwargs,
        )

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        parameters, metrics = aggregated
        if parameters is not None:
            save_authoritative_model(
                self.global_model,
                parameters_to_ndarrays(parameters),
                self.model_path,
            )
        return parameters, metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    weighted_f1 = sum(
        num_examples * float(metric.get("macro_f1", 0.0))
        for num_examples, metric in metrics
    )
    weighted_loss = sum(
        num_examples * float(metric.get("loss", 0.0))
        for num_examples, metric in metrics
    )
    global_f1 = weighted_f1 / total_examples
    global_loss = weighted_loss / total_examples
    round_metrics.append(
        {
            "round": len(round_metrics) + 1,
            "macro_f1": global_f1,
            "loss": global_loss,
        }
    )
    print(
        f"\nGlobal multiclass metrics | "
        f"Macro F1: {global_f1:.4f} | Loss: {global_loss:.4f}\n"
    )
    return {
        "macro_f1": float(global_f1),
        "loss": float(global_loss),
    }


def save_metrics():
    output_path = MODELS_DIR / "flower_history.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(round_metrics, file, indent=2)
    print(f"Metrics saved to {output_path}")


def start_server(num_rounds=15, min_clients=3):
    global_model = load_authoritative_model()
    strategy = MulticlassFedAvg(
        global_model=global_model,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print("\n======================================")
    print("        FEDSHIELD FLOWER SERVER")
    print("======================================")
    print(f"Clients required : {min_clients}")
    print(f"Training rounds  : {num_rounds}")
    print("Model contract   : MultiClassIDS (41 -> 5)")
    print("Server address   : 0.0.0.0:8080")
    print("======================================\n")

    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
        evaluate_final_model(global_model)
    finally:
        save_metrics()


if __name__ == "__main__":
    start_server()
