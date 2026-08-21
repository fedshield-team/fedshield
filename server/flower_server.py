import json
import os
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import Metrics


BASE = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(
    BASE,
    "models"
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


round_metrics = []


def weighted_average(
    metrics: List[Tuple[int, Metrics]]
) -> Metrics:

    if not metrics:
        return {}

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    if total_examples == 0:
        return {}

    weighted_f1 = sum(
        num_examples * float(
            metric.get("f1", 0.0)
        )
        for num_examples, metric in metrics
    )

    weighted_loss = sum(
        num_examples * float(
            metric.get("loss", 0.0)
        )
        for num_examples, metric in metrics
    )

    global_f1 = (
        weighted_f1 /
        total_examples
    )

    global_loss = (
        weighted_loss /
        total_examples
    )

    round_metrics.append(
        {
            "round": len(round_metrics) + 1,
            "f1": global_f1,
            "loss": global_loss
        }
    )

    print(
        f"\n🌐 Global metrics | "
        f"F1: {global_f1:.4f} | "
        f"Loss: {global_loss:.4f}\n"
    )

    return {
        "f1": float(global_f1),
        "loss": float(global_loss)
    }


def save_metrics():

    output_path = os.path.join(
        MODELS_DIR,
        "flower_history.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            round_metrics,
            f,
            indent=2
        )

    print(
        f"Metrics saved to {output_path}"
    )


def start_server(
    num_rounds=15,
    min_clients=3
):

    strategy = fl.server.strategy.FedAvg(

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=min_clients,

        min_evaluate_clients=min_clients,

        min_available_clients=min_clients,

        evaluate_metrics_aggregation_fn=weighted_average
    )

    print(
        "\n======================================"
    )

    print(
        "        FEDSHIELD FLOWER SERVER"
    )

    print(
        "======================================"
    )

    print(
        f"Clients required : {min_clients}"
    )

    print(
        f"Training rounds  : {num_rounds}"
    )

    print(
        "Server address   : 0.0.0.0:8080"
    )

    print(
        "======================================\n"
    )

    try:

        fl.server.start_server(

            server_address="0.0.0.0:8080",

            config=fl.server.ServerConfig(
                num_rounds=num_rounds
            ),

            strategy=strategy
        )

    finally:

        save_metrics()


if __name__ == "__main__":

    start_server()