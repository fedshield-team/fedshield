"""Shared multiclass inference and runtime state for FedShield.

The live capture process and the API both use the same model artifact,
preprocessors, feature order, and version file.  This module keeps API
inference independent from packet sniffing while preserving the existing
live_capture pipeline for real traffic.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import joblib
import numpy as np
import shap
import torch

from model import MultiClassIDS


BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "federated_noniid_model.pth"
SCALER_PATH = BASE / "models" / "scaler_multiclass.pkl"
ENCODERS_PATH = BASE / "models" / "encoders_multiclass.pkl"
VERSION_PATH = BASE / "models" / "model_version.json"
TRAIN_BG_PATH = BASE / "data" / "X_train_mc.npy"

CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
FEATURE_NAMES = [
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


class FedShieldRuntime:
    """Thread-safe model loader with version-aware hot reload."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.model: MultiClassIDS | None = None
        self.scaler = None
        self.encoders = None
        self.explainer = None
        self.model_version = None
        self.load_error = None
        self.reload(force=True)

    @staticmethod
    def read_version():
        try:
            with VERSION_PATH.open(encoding="utf-8") as file:
                return json.load(file).get("version")
        except (OSError, ValueError, TypeError):
            return None

    def reload(self, force: bool = False) -> bool:
        with self._lock:
            version = self.read_version()
            if not force and self.model is not None and version == self.model_version:
                return False

            try:
                required = (MODEL_PATH, SCALER_PATH, ENCODERS_PATH)
                missing = [str(path) for path in required if not path.exists()]
                if missing:
                    raise FileNotFoundError(
                        "Missing FedShield runtime artifacts: " + ", ".join(missing)
                    )

                model = MultiClassIDS(input_dim=41, num_classes=5)
                state = torch.load(
                    MODEL_PATH,
                    map_location="cpu",
                    weights_only=True,
                )
                model.load_state_dict(state)
                model.eval()

                scaler = joblib.load(SCALER_PATH)
                encoders = joblib.load(ENCODERS_PATH)
                required_encoders = {"protocol_type", "service", "flag"}
                missing_encoders = required_encoders - set(encoders.keys())
                if missing_encoders:
                    raise ValueError(
                        "Missing categorical encoders: "
                        + ", ".join(sorted(missing_encoders))
                    )

                self.model = model
                self.scaler = scaler
                self.encoders = encoders
                self.model_version = version
                self.explainer = None
                self.load_error = None
                return True
            except Exception as exc:
                self.load_error = str(exc)
                if force:
                    raise
                return False

    def _ensure_explainer(self):
        if self.explainer is not None:
            return self.explainer
        if not TRAIN_BG_PATH.exists() or self.model is None:
            return None
        try:
            background = np.load(TRAIN_BG_PATH)[:100]
            self.explainer = shap.DeepExplainer(
                self.model,
                torch.as_tensor(background, dtype=torch.float32),
            )
        except Exception as exc:
            self.load_error = f"SHAP unavailable: {exc}"
            self.explainer = None
        return self.explainer

    def predict(self, features, *, scaled: bool = False, explain: bool = True):
        self.reload()
        with self._lock:
            if self.model is None or self.scaler is None:
                raise RuntimeError(self.load_error or "FedShield model is unavailable")

            values = np.asarray(features, dtype=np.float32).reshape(-1)
            if values.size != 41:
                raise ValueError(f"Expected exactly 41 features, got {values.size}")

            scaled_values = (
                values
                if scaled
                else self.scaler.transform(values.reshape(1, -1))[0]
            ).astype(np.float32)
            tensor = torch.as_tensor(
                scaled_values.reshape(1, -1),
                dtype=torch.float32,
            )

            with torch.no_grad():
                probabilities = torch.softmax(self.model(tensor), dim=1)[0]

            pred_class = int(torch.argmax(probabilities).item())
            result = {
                "prediction": CLASS_NAMES[pred_class],
                "class_id": pred_class,
                "confidence": float(probabilities[pred_class].item()),
                "probabilities": {
                    name: float(probabilities[index].item())
                    for index, name in enumerate(CLASS_NAMES)
                },
                "model_version": self.model_version,
                "features_scaled": bool(scaled),
                "feature_count": 41,
            }

            if explain:
                result["shap_features"] = self.top_shap_features(
                    tensor,
                    pred_class,
                )
            else:
                result["shap_features"] = []
            return result

    def top_shap_features(self, tensor, pred_class: int, top_n: int = 5):
        explainer = self._ensure_explainer()
        if explainer is None:
            return []
        try:
            values = explainer.shap_values(tensor)
            if isinstance(values, list):
                class_values = np.asarray(values[pred_class])[0]
            else:
                array = np.asarray(values)
                if array.ndim == 3 and array.shape[-1] == len(CLASS_NAMES):
                    class_values = array[0, :, pred_class]
                elif array.ndim == 3 and array.shape[0] == len(CLASS_NAMES):
                    class_values = array[pred_class, 0, :]
                elif array.ndim == 2:
                    class_values = array[0]
                else:
                    return []

            class_values = np.asarray(class_values).reshape(-1)
            if len(class_values) != len(FEATURE_NAMES):
                return []
            indices = np.argsort(np.abs(class_values))[::-1][:top_n]
            return [
                {
                    "feature": FEATURE_NAMES[index],
                    "shap_value": float(class_values[index]),
                }
                for index in indices
            ]
        except Exception as exc:
            self.load_error = f"SHAP computation failed: {exc}"
            return []

    def status(self):
        with self._lock:
            return {
                "model_loaded": self.model is not None,
                "model_path": str(MODEL_PATH.relative_to(BASE)),
                "model_exists": MODEL_PATH.exists(),
                "scaler_exists": SCALER_PATH.exists(),
                "encoders_exists": ENCODERS_PATH.exists(),
                "feature_count": 41,
                "classes": CLASS_NAMES,
                "model_version": self.model_version,
                "shap_available": self.explainer is not None
                or TRAIN_BG_PATH.exists(),
                "load_error": self.load_error,
            }


runtime = FedShieldRuntime()