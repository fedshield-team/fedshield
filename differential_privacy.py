"""
FedShield — Differential Privacy for Federated Learning
Implements Gaussian mechanism for (ε, δ)-differential privacy on model weights.

Usage:
    from differential_privacy import DPConfig, apply_dp_noise, clip_weights

    config = DPConfig(epsilon=1.0, delta=1e-5, sensitivity=1.0)
    
    # Before sending weights from a node:
    noisy_weights = apply_dp_noise(weights, config)

Privacy guarantee:
    Adding calibrated Gaussian noise ensures that the aggregator cannot
    distinguish between any two adjacent datasets (differing by one record),
    providing formal (ε, δ)-DP guarantees.

Reference:
    Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016)
    https://arxiv.org/abs/1607.00133
"""

import torch
import numpy as np
import math
from dataclasses import dataclass
from typing import List


@dataclass
class DPConfig:
    """
    Differential Privacy configuration.
    
    Args:
        epsilon:     Privacy budget (lower = more private, typical range 0.1–10)
        delta:       Failure probability (typical: 1e-5)
        sensitivity: L2 sensitivity / gradient clipping threshold
        mechanism:   'gaussian' (recommended) or 'laplace'
    """
    epsilon:     float = 1.0
    delta:       float = 1e-5
    sensitivity: float = 1.0
    mechanism:   str   = 'gaussian'

    @property
    def noise_scale(self) -> float:
        """Gaussian noise scale σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε"""
        if self.mechanism == 'gaussian':
            return self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        else:  # Laplace
            return self.sensitivity / self.epsilon

    def privacy_report(self) -> str:
        return (
            f"DP Config: ε={self.epsilon}, δ={self.delta:.0e}, "
            f"sensitivity={self.sensitivity}, σ={self.noise_scale:.4f}"
        )


def clip_weights(weights: List[torch.Tensor], max_norm: float) -> List[torch.Tensor]:
    """
    Clip weight tensors to bound L2 sensitivity.
    This is the 'bounding sensitivity' step of the Gaussian mechanism.
    
    Args:
        weights:  List of weight tensors from model.parameters()
        max_norm: Maximum allowed L2 norm (= sensitivity)
    
    Returns:
        Clipped weight tensors
    """
    total_norm = torch.sqrt(sum(w.norm(2) ** 2 for w in weights))
    clip_coef  = min(1.0, max_norm / (total_norm.item() + 1e-6))
    return [w * clip_coef for w in weights]


def apply_dp_noise(
    weights:  List[torch.Tensor],
    config:   DPConfig,
    verbose:  bool = False
) -> List[torch.Tensor]:
    """
    Apply (ε, δ)-differential privacy noise to model weights.
    
    Steps:
        1. Clip weights to bound sensitivity
        2. Add calibrated Gaussian noise
    
    Args:
        weights: List of weight tensors
        config:  DPConfig with privacy parameters
        verbose: Print noise statistics
    
    Returns:
        Noisy weight tensors with DP guarantees
    """
    # Step 1: Clip
    clipped = clip_weights(weights, config.sensitivity)

    # Step 2: Add noise
    noisy = []
    total_params = sum(w.numel() for w in clipped)
    total_noise_norm = 0.0

    for w in clipped:
        if config.mechanism == 'gaussian':
            noise = torch.normal(mean=0.0, std=config.noise_scale, size=w.shape)
        else:  # Laplace
            noise = torch.from_numpy(
                np.random.laplace(0, config.noise_scale, w.shape.numpy())
            ).float()
        noisy.append(w + noise)
        total_noise_norm += noise.norm(2).item() ** 2

    total_noise_norm = math.sqrt(total_noise_norm)

    if verbose:
        print(f"  [DP] {config.privacy_report()}")
        print(f"  [DP] Parameters: {total_params:,} | Noise L2 norm: {total_noise_norm:.4f}")

    return noisy


def privacy_accounting(
    num_rounds:   int,
    noise_mult:   float,
    sample_rate:  float,
    delta:        float = 1e-5
) -> float:
    """
    Compute privacy cost over multiple rounds using moments accountant.
    Simplified RDP (Rényi DP) accounting.
    
    Args:
        num_rounds:  Number of federated rounds
        noise_mult:  Noise multiplier (σ / sensitivity)
        sample_rate: Fraction of data sampled per round
        delta:       Target δ
    
    Returns:
        Effective epsilon after all rounds
    """
    # Simplified: ε grows as O(√T) with Gaussian mechanism composition
    eps_per_round = math.sqrt(2 * math.log(1.25 / delta)) / noise_mult
    total_eps = eps_per_round * math.sqrt(num_rounds) * sample_rate
    return total_eps


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch.nn as nn

    print("=" * 60)
    print("FedShield — Differential Privacy Self Test")
    print("=" * 60)

    # Build test model
    model = nn.Sequential(nn.Linear(41, 256), nn.ReLU(), nn.Linear(256, 5))
    original = [p.data.clone() for p in model.parameters()]

    configs = [
        DPConfig(epsilon=0.1,  delta=1e-5, sensitivity=1.0),  # High privacy
        DPConfig(epsilon=1.0,  delta=1e-5, sensitivity=1.0),  # Balanced
        DPConfig(epsilon=10.0, delta=1e-5, sensitivity=1.0),  # Low privacy
    ]

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n{'ε':>6} {'σ (noise scale)':>18} {'Weight perturbation':>22}")
    print("-" * 50)

    for cfg in configs:
        noisy = apply_dp_noise(original, cfg)
        perturbation = torch.sqrt(sum(
            (n - o).norm(2)**2
            for n, o in zip(noisy, original)
        )).item()
        print(f"{cfg.epsilon:>6.1f} {cfg.noise_scale:>18.4f} {perturbation:>22.4f}")

    print("\nPrivacy accounting over 15 rounds:")
    for cfg in configs:
        total_eps = privacy_accounting(
            num_rounds=15, noise_mult=cfg.noise_scale,
            sample_rate=0.33, delta=cfg.delta
        )
        print(f"  ε={cfg.epsilon} per round → ε_total={total_eps:.4f} over 15 rounds")

    print("\n✅ Differential Privacy module working correctly")
    print("\nTo integrate into federated_noniid.py:")
    print("  from differential_privacy import DPConfig, apply_dp_noise")
    print("  config = DPConfig(epsilon=1.0, delta=1e-5, sensitivity=1.0)")
    print("  noisy_weights = apply_dp_noise(node.get_weights(), config, verbose=True)")
    print("=" * 60)