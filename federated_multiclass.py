"""
FedShield — Multi-Class Federated Training

The project now uses federated_noniid.py as the
single authoritative federated training pipeline.

This file is kept as a compatibility entry point.
"""

import runpy
import os


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TARGET = os.path.join(
    BASE_DIR,
    "federated_noniid.py"
)


if __name__ == "__main__":

    runpy.run_path(
        TARGET,
        run_name="__main__"
    )