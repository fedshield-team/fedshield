import flwr as fl
import torch
import json
from typing import List, Tuple, Optional, Dict
from flwr.common import Metrics
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import IntrusionDetector

# Track metrics across rounds
round_metrics = []

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    f1_scores = [m["f1"] * n for n, m in metrics]
    total = sum(n for n, _ in metrics)
    avg_f1 = sum(f1_scores) / total
    round_metrics.append({"f1": avg_f1})
    print(f"\n🌐 Global F1 after aggregation: {avg_f1:.4f}\n")
    return {"f1": avg_f1}

def save_metrics():
    with open("models/flower_history.json", "w") as f:
        json.dump([{"round": i+1, "f1": m["f1"]} 
                   for i, m in enumerate(round_metrics)], f)
    print("Metrics saved to models/flower_history.json")

def start_server(num_rounds=10, min_clients=3):
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print(f"🚀 FedShield Server starting...")
    print(f"   Waiting for {min_clients} clients...")
    print(f"   Rounds: {num_rounds}\n")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    save_metrics()

if __name__ == "__main__":
    start_server()