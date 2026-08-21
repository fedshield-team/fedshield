import os

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


COLUMNS = [
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
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty"
]


def load_and_preprocess(
    path
):

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS
    )

    df.drop(
        columns=["difficulty"],
        inplace=True
    )

    categorical_columns = [
        "protocol_type",
        "service",
        "flag"
    ]

    encoders = {}

    for column in categorical_columns:

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(
            df[column].astype(str)
        )

        encoders[column] = encoder

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .apply(
            lambda value:
                0
                if value == "normal"
                else 1
        )
    )

    X = df.drop(
        columns=["label"]
    ).values

    y = df["label"].values


    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )


    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )


    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoders
    )


if __name__ == "__main__":

    input_path = os.path.join(
        DATA_DIR,
        "KDDTrain+.txt"
    )

    print(
        "===== NSL-KDD BINARY PREPROCESSING ====="
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoders
    ) = load_and_preprocess(
        input_path
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Test samples: {len(X_test)}"
    )

    print(
        f"Features: {X_train.shape[1]}"
    )

    print(
        f"Attack ratio: "
        f"{y_train.mean():.2%}"
    )


    np.save(
        os.path.join(
            DATA_DIR,
            "X_train.npy"
        ),
        X_train
    )

    np.save(
        os.path.join(
            DATA_DIR,
            "X_test.npy"
        ),
        X_test
    )

    np.save(
        os.path.join(
            DATA_DIR,
            "y_train.npy"
        ),
        y_train
    )

    np.save(
        os.path.join(
            DATA_DIR,
            "y_test.npy"
        ),
        y_test
    )


    joblib.dump(
        scaler,
        os.path.join(
            MODELS_DIR,
            "scaler_binary.pkl"
        )
    )

    joblib.dump(
        encoders,
        os.path.join(
            MODELS_DIR,
            "encoders_binary.pkl"
        )
    )


    print(
        "\nSaved:"
    )

    print(
        "  data/X_train.npy"
    )

    print(
        "  data/X_test.npy"
    )

    print(
        "  data/y_train.npy"
    )

    print(
        "  data/y_test.npy"
    )

    print(
        "  models/scaler_binary.pkl"
    )

    print(
        "  models/encoders_binary.pkl"
    )