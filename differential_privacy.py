"""FedShield — Differential Privacy utilities.

Important: this module implements clipped model-output perturbation. A formal
end-to-end federated DP guarantee also depends on how client updates are
formed, sampled, aggregated, and composed across rounds.
"""

from dataclasses import dataclass
import math
from typing import List

import numpy as np
import torch


@dataclass
class DPConfig:
    epsilon: float = 1.0
    delta: float = 1e-5
    sensitivity: float = 1.0
    mechanism: str = "gaussian"

    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")

        if not 0 < self.delta < 1:
            raise ValueError("delta must be between 0 and 1")

        if self.sensitivity <= 0:
            raise ValueError("sensitivity must be > 0")

        self.mechanism = self.mechanism.lower()

        if self.mechanism not in {"gaussian", "laplace"}:
            raise ValueError(
                "mechanism must be 'gaussian' or 'laplace'"
            )

    @property
    def noise_scale(self) -> float:
        if self.mechanism == "gaussian":
            return (
                self.sensitivity
                * math.sqrt(
                    2 * math.log(1.25 / self.delta)
                )
                / self.epsilon
            )

        return self.sensitivity / self.epsilon

    def privacy_report(self) -> str:
        return (
            f"DP Config: epsilon={self.epsilon}, "
            f"delta={self.delta:.0e}, "
            f"sensitivity={self.sensitivity}, "
            f"sigma={self.noise_scale:.4f}"
        )


def clip_weights(
    weights: List[torch.Tensor],
    max_norm: float
) -> List[torch.Tensor]:
    """Clip the concatenated model-weight vector to max_norm."""

    if max_norm <= 0:
        raise ValueError("max_norm must be > 0")

    tensors = [w.detach() for w in weights]

    if not tensors:
        return []

    total_sq = sum(
        torch.sum(w.float() ** 2)
        for w in tensors
    )

    total_norm = torch.sqrt(total_sq).item()

    coef = min(
        1.0,
        max_norm / (total_norm + 1e-12)
    )

    return [w * coef for w in tensors]


def apply_dp_noise(
    weights: List[torch.Tensor],
    config: DPConfig,
    verbose: bool = False
) -> List[torch.Tensor]:
    """Clip weights and add independent noise to each tensor.

    For a client-level Gaussian mechanism where two clipped outputs can differ
    by at most 2*C, pass sensitivity=2*C.

    If your adjacency definition differs, derive sensitivity from that
    definition before claiming a formal DP bound.
    """

    clipped = clip_weights(
        weights,
        config.sensitivity
    )

    noisy = []

    noise_sq = 0.0
    total_params = 0

    for w in clipped:

        if config.mechanism == "gaussian":

            noise = (
                torch.randn(
                    w.shape,
                    dtype=w.dtype,
                    device=w.device
                )
                * config.noise_scale
            )

        else:

            arr = np.random.laplace(
                0.0,
                config.noise_scale,
                size=tuple(w.shape)
            )

            noise = torch.as_tensor(
                arr,
                dtype=w.dtype,
                device=w.device
            )

        noisy.append(w + noise)

        total_params += w.numel()

        noise_sq += float(
            torch.sum(
                noise.float() ** 2
            ).item()
        )

    if verbose:
        print(
            f"  [DP] {config.privacy_report()}"
        )

        print(
            f"  [DP] Parameters: "
            f"{total_params:,} | "
            f"Noise L2 norm: "
            f"{math.sqrt(noise_sq):.4f}"
        )

    return noisy


def privacy_accounting(
    num_rounds: int,
    noise_mult: float,
    sample_rate: float,
    delta: float = 1e-5
) -> float:
    """Return a conservative Gaussian-composition approximation.

    This is intentionally named as an approximation, not a formal RDP
    accountant.

    Use a formal RDP accountant such as Opacus for publishable
    privacy accounting.
    """

    if (
        num_rounds < 1
        or noise_mult <= 0
        or not 0 < sample_rate <= 1
        or not 0 < delta < 1
    ):
        raise ValueError(
            "invalid privacy-accounting parameters"
        )

    eps_per_round = (
        math.sqrt(
            2 * math.log(1.25 / delta)
        )
        / noise_mult
    )

    return (
        eps_per_round
        * math.sqrt(num_rounds)
        * sample_rate
    )


if __name__ == "__main__":

    import torch.nn as nn

    print("=" * 60)
    print("FedShield — Differential Privacy Self Test")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(41, 256),
        nn.ReLU(),
        nn.Linear(256, 5)
    )

    original = [
        p.detach().clone()
        for p in model.parameters()
    ]

    configs = [
        DPConfig(0.1),
        DPConfig(1.0),
        DPConfig(10.0)
    ]

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    for cfg in configs:

        noisy = apply_dp_noise(
            original,
            cfg
        )

        perturbation = torch.sqrt(
            sum(
                torch.sum(
                    (n - o).float() ** 2
                )
                for n, o in zip(
                    noisy,
                    original
                )
            )
        ).item()

        print(
            f"epsilon={cfg.epsilon:>4.1f}  "
            f"sigma={cfg.noise_scale:>10.4f}  "
            f"perturbation={perturbation:>10.4f}"
        )

    print(
        "\nNote: use a formal RDP accountant "
        "before reporting an end-to-end "
        "multi-round epsilon."
    )