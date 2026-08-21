import torch
import torch.nn as nn


class IntrusionDetector(nn.Module):
    """
    Binary Intrusion Detection Model.

    Input:
        Network traffic features

    Output:
        Probability of:
        0 = Normal
        1 = Attack
    """

    def __init__(self, input_dim=41):
        super().__init__()

        self.input_dim = input_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

    def get_weights(self):
        """
        Return model parameters only.
        Used by the custom PyTorch federated trainer.
        """
        return [
            parameter.detach().clone()
            for parameter in self.parameters()
        ]

    def set_weights(self, weights):
        """
        Load model parameters.
        """
        if len(weights) != len(list(self.parameters())):
            raise ValueError(
                f"Expected {len(list(self.parameters()))} tensors, "
                f"received {len(weights)}"
            )

        with torch.no_grad():
            for parameter, weight in zip(self.parameters(), weights):
                parameter.copy_(weight)