import os
import sys

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from model import IntrusionDetector


class FedShieldClient(fl.client.NumPyClient):

    def __init__(
        self,
        node_id,
        X_train,
        y_train,
        X_test,
        y_test,
        input_dim=None,
        lr=0.001,
        local_epochs=3
    ):

        self.node_id = node_id

        if input_dim is None:
            input_dim = X_train.shape[1]

        self.input_dim = input_dim
        self.lr = lr
        self.local_epochs = local_epochs

        self.model = IntrusionDetector(
            input_dim=input_dim
        )

        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        X_train_t = torch.as_tensor(
            X_train,
            dtype=torch.float32
        )

        y_train_t = torch.as_tensor(
            y_train,
            dtype=torch.float32
        ).view(-1, 1)

        self.loader = DataLoader(
            TensorDataset(
                X_train_t,
                y_train_t
            ),
            batch_size=256,
            shuffle=True
        )

        self.X_test = torch.as_tensor(
            X_test,
            dtype=torch.float32
        )

        self.y_test = torch.as_tensor(
            y_test,
            dtype=torch.float32
        ).view(-1)

        print(
            f"[Node {node_id}] Ready | "
            f"Train: {len(X_train)} | "
            f"Features: {input_dim}"
        )

    def get_parameters(self, config):

        return [
            parameter.detach().cpu().numpy()
            for parameter in self.model.parameters()
        ]

    def set_parameters(self, parameters):

        if len(parameters) != len(
            list(self.model.parameters())
        ):
            raise ValueError(
                "Received incorrect number of model tensors"
            )

        with torch.no_grad():

            for parameter, new_parameter in zip(
                self.model.parameters(),
                parameters
            ):

                parameter.copy_(
                    torch.as_tensor(
                        new_parameter,
                        dtype=parameter.dtype
                    )
                )

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        # Reset optimizer state whenever a new global model arrives.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

        self.model.train()

        total_loss = 0.0
        batches = 0

        for _ in range(self.local_epochs):

            for X_batch, y_batch in self.loader:

                self.optimizer.zero_grad()

                logits = self.model(X_batch)

                loss = self.criterion(
                    logits,
                    y_batch
                )

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batches += 1

        avg_loss = total_loss / max(batches, 1)

        print(
            f"[Node {self.node_id}] "
            f"Training complete | "
            f"Loss: {avg_loss:.4f}"
        )

        return (
            self.get_parameters(config),
            len(self.loader.dataset),
            {
                "loss": float(avg_loss)
            }
        )

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        self.model.eval()

        with torch.no_grad():

            logits = self.model(self.X_test)

            loss = self.criterion(
                logits,
                self.y_test.unsqueeze(1)
            ).item()

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
            ).long().view(-1)

        f1 = f1_score(
            self.y_test.long().numpy(),
            predictions.numpy(),
            zero_division=0
        )

        print(
            f"[Node {self.node_id}] "
            f"Evaluation F1: {f1:.4f}"
        )

        return (
            float(loss),
            len(self.X_test),
            {
                "f1": float(f1)
            }
        )


def start_client(node_id, server_address=None):

    if server_address is None:

        server_address = os.environ.get(
            "SERVER_ADDRESS",
            "server:8080"
        )

    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    data_dir = os.path.join(
        base_dir,
        "data"
    )

    X_train = np.load(
        os.path.join(data_dir, "X_train.npy")
    )

    y_train = np.load(
        os.path.join(data_dir, "y_train.npy")
    )

    X_test = np.load(
        os.path.join(data_dir, "X_test.npy")
    )

    y_test = np.load(
        os.path.join(data_dir, "y_test.npy")
    )

    input_dim = X_train.shape[1]

    n = len(X_train)

    split = n // 3

    splits = {
        1: (
            X_train[:split],
            y_train[:split]
        ),

        2: (
            X_train[split:2 * split],
            y_train[split:2 * split]
        ),

        3: (
            X_train[2 * split:],
            y_train[2 * split:]
        )
    }

    if node_id not in splits:
        raise ValueError(
            "node_id must be 1, 2, or 3"
        )

    X, y = splits[node_id]

    client = FedShieldClient(
        node_id=node_id,
        X_train=X,
        y_train=y,
        X_test=X_test,
        y_test=y_test,
        input_dim=input_dim
    )

    print(
        f"[Node {node_id}] "
        f"Connecting to Flower server at "
        f"{server_address}"
    )

    fl.client.start_client(
        server_address=server_address,
        client=client.to_client()
    )


if __name__ == "__main__":

    node_id = (
        int(sys.argv[1])
        if len(sys.argv) > 1
        else 1
    )

    start_client(node_id)