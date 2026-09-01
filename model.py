import torch
import torch.nn as nn


# ============================================================
# FEDSHIELD MULTI-CLASS MODEL
# ============================================================

class MultiClassIDS(nn.Module):

    def __init__(
        self,
        input_dim=41,
        num_classes=5
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):

        return self.network(x)

    def get_weights(self):

        return [
            parameter.detach().clone()
            for parameter in self.parameters()
        ]

    def set_weights(self, weights):

        parameters = list(
            self.parameters()
        )

        if len(parameters) != len(weights):
            raise ValueError(
                "Number of model tensors does not match."
            )

        with torch.no_grad():

            for parameter, weight in zip(
                parameters,
                weights
            ):

                if parameter.shape != weight.shape:
                    raise ValueError(
                        "Model tensor shape mismatch: "
                        f"{parameter.shape} != "
                        f"{weight.shape}"
                    )

                parameter.copy_(
                    weight.to(
                        dtype=parameter.dtype
                    )
                )


# ============================================================
# OLD BINARY MODEL
#
# Kept so the existing binary pipeline does not break.
# ============================================================

class IntrusionDetector(nn.Module):

    def __init__(
        self,
        input_dim=41
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.network(x)

    def get_weights(self):

        return [
            parameter.detach().clone()
            for parameter in self.parameters()
        ]

    def set_weights(self, weights):

        parameters = list(
            self.parameters()
        )

        if len(parameters) != len(weights):
            raise ValueError(
                "Number of model tensors does not match."
            )

        with torch.no_grad():

            for parameter, weight in zip(
                parameters,
                weights
            ):

                if parameter.shape != weight.shape:
                    raise ValueError(
                        "Model tensor shape mismatch."
                    )

                parameter.copy_(
                    weight.to(
                        dtype=parameter.dtype
                    )
                )