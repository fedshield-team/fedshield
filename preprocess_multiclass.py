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


ATTACK_MAP = {

    "normal": 0,

    "neptune": 1,
    "back": 1,
    "teardrop": 1,
    "pod": 1,
    "smurf": 1,
    "land": 1,
    "mailbomb": 1,
    "apache2": 1,

    "satan": 2,
    "ipsweep": 2,
    "portsweep": 2,
    "nmap": 2,
    "mscan": 2,

    "warezclient": 3,
    "warezmaster": 3,
    "imap": 3,
    "ftp_write": 3,
    "multihop": 3,
    "guess_passwd": 3,
    "phf": 3,
    "spy": 3,
    "sendmail": 3,

    "buffer_overflow": 4,
    "rootkit": 4,
    "loadmodule": 4,
    "perl": 4,
    "xterm": 4
}


CLASS_NAMES = [
    "Normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R"
]


def load_multiclass(
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


    labels = (
        df["label"]
        .astype(str)
        .str.strip()
        .map(ATTACK_MAP)
    )


    valid = labels.notna()

    df = df.loc[
        valid
    ].copy()

    df["label"] = (
        labels.loc[valid]
        .astype(int)
    )


    print(
        "Class distribution:"
    )

    for index, name in enumerate(
        CLASS_NAMES
    ):

        count = int(
            (df["label"] == index)
            .sum()
        )

        print(
            f"  {name}: {count}"
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
        "===== NSL-KDD MULTI-CLASS PREPROCESSING ====="
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoders
    ) = load_multiclass(
        input_path
    )


    np.save(
        os.path.join(
            DATA_DIR,
            "X_train_mc.npy"
        ),
        X_train
    )

    np.save(
        os.path.join(
            DATA_DIR,
            "X_test_mc.npy"
        ),
        X_test
    )

    np.save(
        os.path.join(
            DATA_DIR,
            "y_train_mc.npy"
        ),
        y_train
    )

    np.save(
        os.path.join(
            DATA_DIR,
            "y_test_mc.npy"
        ),
        y_test
    )


    joblib.dump(
        scaler,
        os.path.join(
            MODELS_DIR,
            "scaler_multiclass.pkl"
        )
    )

    joblib.dump(
        encoders,
        os.path.join(
            MODELS_DIR,
            "encoders_multiclass.pkl"
        )
    )


    print(
        "\nSaved multi-class dataset:"
    )

    print(
        "  data/X_train_mc.npy"
    )

    print(
        "  data/X_test_mc.npy"
    )

    print(
        "  data/y_train_mc.npy"
    )

    print(
        "  data/y_test_mc.npy"
    )

    print(
        "  models/scaler_multiclass.pkl"
    )

    print(
        "  models/encoders_multiclass.pkl"
    )