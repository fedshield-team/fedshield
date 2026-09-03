"""
FedShield — Offline SHAP Analysis

Uses the exact same MultiClassIDS architecture
as federated training and live inference.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import torch
import shap

from model import (
    MULTICLASS_CLASS_NAMES,
    MULTICLASS_FEATURE_NAMES,
    MULTICLASS_INPUT_DIM,
    MULTICLASS_NUM_CLASSES,
    MultiClassIDS,
)


# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE,
    "models",
    "federated_noniid_model.pth"
)

X_TEST_PATH = os.path.join(
    BASE,
    "data",
    "X_test_mc.npy"
)

VERSION_PATH = os.path.join(
    BASE,
    "models",
    "model_version.json"
)

OUTPUT_PATH = os.path.join(
    BASE,
    "models",
    "shap_results.json"
)


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = MULTICLASS_CLASS_NAMES
FEATURE_NAMES = MULTICLASS_FEATURE_NAMES


# ============================================================
# MAIN
# ============================================================

def main():

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}\n\n"
            "Run federated_noniid.py first."
        )

    if not os.path.exists(
        X_TEST_PATH
    ):

        raise FileNotFoundError(
            f"Test data not found:\n"
            f"{X_TEST_PATH}\n\n"
            "Run preprocess_multiclass.py first."
        )

    try:
        with open(
            VERSION_PATH,
            encoding="utf-8"
        ) as f:
            model_version = json.load(f).get("version")
    except (
        OSError,
        ValueError,
        TypeError
    ) as e:
        raise RuntimeError(
            f"Model version metadata unavailable: {e}"
        ) from e

    if model_version is None:
        raise RuntimeError(
            "Model version metadata does not contain a version."
        )

    # --------------------------------------------------------
    # LOAD SAME MODEL
    # --------------------------------------------------------

    model = MultiClassIDS(
        input_dim=MULTICLASS_INPUT_DIM,
        num_classes=MULTICLASS_NUM_CLASSES
    )

    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        state
    )

    model.eval()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    X_test = np.load(
        X_TEST_PATH
    )

    if (
        X_test.ndim != 2
        or
        X_test.shape[1] != 41
    ):

        raise ValueError(
            f"Expected shape "
            f"(samples, 41), "
            f"got {X_test.shape}"
        )

    # Small SHAP dataset.
    background = X_test[:50]

    samples = X_test[
        50:100
    ]

    if len(samples) == 0:

        raise ValueError(
            "Not enough test samples for SHAP."
        )

    # --------------------------------------------------------
    # PREDICTION FUNCTION
    # --------------------------------------------------------

    def predict(x):

        tensor = torch.as_tensor(
            x,
            dtype=torch.float32
        )

        with torch.no_grad():

            logits = model(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

        return probabilities.cpu().numpy()

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    print(
        "Computing SHAP values..."
    )

    explainer = shap.KernelExplainer(
        predict,
        background
    )

    values = explainer.shap_values(
        samples,
        nsamples=100
    )

    # --------------------------------------------------------
    # NORMALIZE SHAP OUTPUT
    # --------------------------------------------------------

    if isinstance(
        values,
        list
    ):

        class_arrays = [
            np.asarray(
                value
            )
            for value in values
        ]

        stacked = np.stack(
            class_arrays,
            axis=-1
        )

        mean_abs = np.abs(
            stacked
        ).mean(
            axis=(0, 2)
        )

    else:

        arr = np.asarray(
            values
        )

        if arr.ndim == 2:

            mean_abs = np.abs(
                arr
            ).mean(
                axis=0
            )

        elif (
            arr.ndim == 3
            and
            arr.shape[1] == 41
        ):

            mean_abs = np.abs(
                arr
            ).mean(
                axis=(0, 2)
            )

        elif (
            arr.ndim == 3
            and
            arr.shape[2] == 41
        ):

            mean_abs = np.abs(
                arr
            ).mean(
                axis=(0, 1)
            )

        else:

            raise ValueError(
                f"Unsupported SHAP shape: "
                f"{arr.shape}"
            )

    if len(mean_abs) != MULTICLASS_INPUT_DIM:

        raise ValueError(
            f"Expected 41 SHAP scores, "
            f"got {len(mean_abs)}"
        )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    importance = sorted(
        zip(
            FEATURE_NAMES,
            mean_abs.tolist()
        ),
        key=lambda item: item[1],
        reverse=True
    )

    print(
        "\n===== TOP 10 FEATURES ====="
    )

    for i, (
        name,
        score
    ) in enumerate(
        importance[:10],
        1
    ):

        print(
            f"{i:2d}. "
            f"{name:35s} "
            f"SHAP: {score:.4f}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "model":
                    "federated_noniid_model",

                "model_version":
                    model_version,

                "model_path":
                    os.path.relpath(
                        MODEL_PATH,
                        BASE
                    ),

                "generated_at":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "classes":
                    CLASS_NAMES,

                "feature_count":
                    MULTICLASS_INPUT_DIM,

                "feature_importance":
                    [
                        {
                            "feature":
                                name,

                            "shap_score":
                                float(score)
                        }

                        for name, score
                        in importance
                    ]
            },
            f,
            indent=2
        )

    print(
        f"\nSHAP results saved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":

    main()