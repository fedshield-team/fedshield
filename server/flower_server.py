"""Flower server for the authoritative FedShield multiclass model path."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import flwr as fl
import torch
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays

from model import (
    MULTICLASS_INPUT_DIM,
    MULTICLASS_NUM_CLASSES,
    MultiClassIDS,
)


BASE = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE / "models"
MODEL_PATH = MODELS_DIR / "federated_noniid_model.pth"
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
    torch.save(model.state_dict(), model_path)


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
    finally:
        save_metrics()


if __name__ == "__main__":
    start_server()