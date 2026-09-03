import torch
import torch.nn as nn


# ============================================================
# FEDSHIELD MULTI-CLASS CONTRACT
# ============================================================

MULTICLASS_INPUT_DIM = 41
MULTICLASS_NUM_CLASSES = 5
MULTICLASS_CLASS_NAMES = [
    "Normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R",
]
MULTICLASS_FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]


# ============================================================
# FEDSHIELD MULTI-CLASS MODEL
# ============================================================

class MultiClassIDS(nn.Module):

    def __init__(
        self,
        input_dim=MULTICLASS_INPUT_DIM,
        num_classes=MULTICLASS_NUM_CLASSES
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

    def predict_proba(self, x):
        """Return probabilities while keeping logits for BCEWithLogitsLoss."""
        return torch.sigmoid(self.forward(x))

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