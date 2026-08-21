"""FedShield — offline SHAP feature-importance analysis."""

import json
import os

import numpy as np
import torch
import shap

from model import IntrusionDetector


BASE = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE,
    "models",
    "federated_model.pth"
)

X_TEST_PATH = os.path.join(
    BASE,
    "data",
    "X_test.npy"
)

OUTPUT_PATH = os.path.join(
    BASE,
    "models",
    "shap_results.json"
)


FEATURE_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]


def main():

    model = IntrusionDetector()

    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        state
    )

    model.eval()

    X_test = np.load(
        X_TEST_PATH
    )

    if (
        X_test.ndim != 2
        or
        X_test.shape[1]
        != len(FEATURE_NAMES)
    ):

        raise ValueError(
            f"Expected X_test with "
            f"{len(FEATURE_NAMES)} features, "
            f"got {X_test.shape}"
        )

    background = torch.as_tensor(
        X_test[:100],
        dtype=torch.float32
    )

    samples = torch.as_tensor(
        X_test[100:200],
        dtype=torch.float32
    )

    def predict(x):

        with torch.no_grad():

            return model(
                torch.as_tensor(
                    x,
                    dtype=torch.float32
                )
            ).cpu().numpy()

    print(
        "Computing SHAP values..."
    )

    explainer = shap.KernelExplainer(
        predict,
        background.numpy()
    )

    values = explainer.shap_values(
        samples.numpy(),
        nsamples=100
    )

    if isinstance(values, list):

        arr = np.stack(
            [
                np.asarray(v)
                for v in values
            ],
            axis=-1
        )

        mean_abs = np.abs(arr).mean(
            axis=(0, 2)
        )

    else:

        arr = np.asarray(values)

        if arr.ndim == 2:

            mean_abs = np.abs(
                arr
            ).mean(axis=0)

        elif (
            arr.ndim == 3
            and
            arr.shape[1]
            == len(FEATURE_NAMES)
        ):

            mean_abs = np.abs(
                arr
            ).mean(axis=(0, 2))

        elif (
            arr.ndim == 3
            and
            arr.shape[2]
            == len(FEATURE_NAMES)
        ):

            mean_abs = np.abs(
                arr
            ).mean(axis=(0, 1))

        else:

            raise ValueError(
                f"Unsupported SHAP output shape: "
                f"{arr.shape}"
            )

    importance = sorted(
        zip(
            FEATURE_NAMES,
            mean_abs.tolist()
        ),
        key=lambda x: x[1],
        reverse=True
    )

    print(
        "\n===== TOP 10 FEATURES ====="
    )

    for i, (name, score) in enumerate(
        importance[:10],
        1
    ):

        print(
            f"{i:2d}. "
            f"{name:35s} "
            f"SHAP: {score:.4f}"
        )

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
                "feature_importance": [
                    {
                        "feature": name,
                        "shap_score": float(score)
                    }
                    for name, score
                    in importance
                ]
            },
            f,
            indent=2
        )

    print(
        f"\nSHAP results saved to "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()