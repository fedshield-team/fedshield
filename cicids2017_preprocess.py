import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "cicids2017"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


LABEL_MAP = {

    "BENIGN": "Normal",

    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DoS",
    "Heartbleed": "DoS",

    "PortScan": "Probe",

    "FTP-Patator": "R2L",
    "SSH-Patator": "R2L",
    "Web Attack - Brute Force": "R2L",
    "Web Attack - XSS": "R2L",
    "Web Attack - Sql Injection": "R2L",

    "Infiltration": "U2R",
    "Bot": "U2R"
}


CLASS_ORDER = [
    "Normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R"
]


CLASS_TO_ID = {
    name: index
    for index, name
    in enumerate(CLASS_ORDER)
}


# 41 network-level features.
SELECTED_FEATURES = [

    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size"
]


def normalize_column_name(
    name
):

    return (
        str(name)
        .replace("\ufeff", "")
        .strip()
    )


def normalize_label(
    label
):

    if pd.isna(label):
        return None

    value = (
        str(label)
        .strip()
        .replace("–", "-")
        .replace("—", "-")
    )

    aliases = {

        "Web Attack - Brute Force":
            "Web Attack - Brute Force",

        "Web Attack - XSS":
            "Web Attack - XSS",

        "Web Attack - Sql Injection":
            "Web Attack - Sql Injection"
    }

    return aliases.get(
        value,
        value
    )


def load_cicids2017(
    data_dir
):

    if not os.path.exists(
        data_dir
    ):

        raise FileNotFoundError(
            f"Directory not found: "
            f"{data_dir}"
        )


    csv_files = sorted(
        file
        for file in os.listdir(
            data_dir
        )
        if file.lower().endswith(
            ".csv"
        )
    )


    if not csv_files:

        raise FileNotFoundError(
            "No CICIDS2017 CSV files found.\n"
            f"Place the dataset CSV files in:\n"
            f"{data_dir}"
        )


    frames = []


    for filename in csv_files:

        path = os.path.join(
            data_dir,
            filename
        )

        print(
            f"Loading {filename}..."
        )

        try:

            frame = pd.read_csv(
                path,
                encoding="latin-1",
                low_memory=False
            )

            frames.append(
                frame
            )

        except Exception as exc:

            print(
                f"[WARN] "
                f"Could not load {filename}: "
                f"{exc}"
            )


    if not frames:

        raise RuntimeError(
            "No CICIDS2017 files "
            "could be loaded."
        )


    combined = pd.concat(
        frames,
        ignore_index=True
    )


    print(
        f"\nTotal rows: "
        f"{len(combined):,}"
    )


    return combined


def preprocess(
    df
):

    # Normalize column names.
    df.columns = [
        normalize_column_name(
            column
        )
        for column in df.columns
    ]


    if "Label" not in df.columns:

        raise ValueError(
            "CICIDS2017 Label column "
            "was not found."
        )


    df["Label"] = (
        df["Label"]
        .apply(normalize_label)
    )


    df["Label"] = (
        df["Label"]
        .map(LABEL_MAP)
    )


    df = df.dropna(
        subset=["Label"]
    ).copy()


    available_features = [

        feature

        for feature in SELECTED_FEATURES

        if feature in df.columns
    ]


    missing_features = [

        feature

        for feature in SELECTED_FEATURES

        if feature not in df.columns
    ]


    print(
        f"\nFeatures available: "
        f"{len(available_features)}/"
        f"{len(SELECTED_FEATURES)}"
    )


    if missing_features:

        print(
            "\nMissing features:"
        )

        for feature in missing_features:

            print(
                f"  - {feature}"
            )


    if len(available_features) < 30:

        raise ValueError(
            "Too many CICIDS2017 features "
            "are missing."
        )


    X = df[
        available_features
    ].copy()


    y = df["Label"].map(
        CLASS_TO_ID
    ).astype(int)


    # Convert all selected features to numeric.
    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )


    # Remove invalid values.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )


    X = X.fillna(
        X.median(
            numeric_only=True
        )
    )


    # Any columns that remain entirely NaN
    # are replaced with zero.
    X = X.fillna(0)


    X = X.to_numpy(
        dtype=np.float32
    )

    y = y.to_numpy(
        dtype=np.int64
    )


    print(
        "\nClass distribution:"
    )


    for class_id, class_name in enumerate(
        CLASS_ORDER
    ):

        count = int(
            np.sum(
                y == class_id
            )
        )

        print(
            f"  {class_name:8}: "
            f"{count:,}"
        )


    # Split BEFORE fitting scaler.
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
    ).astype(np.float32)

    X_test = scaler.transform(
        X_test
    ).astype(np.float32)


    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        available_features
    )


def save_artifacts(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    features
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


    np.save(
        os.path.join(
            MODEL_DIR,
            "X_train_cicids2017.npy"
        ),
        X_train
    )

    np.save(
        os.path.join(
            MODEL_DIR,
            "X_test_cicids2017.npy"
        ),
        X_test
    )

    np.save(
        os.path.join(
            MODEL_DIR,
            "y_train_cicids2017.npy"
        ),
        y_train
    )

    np.save(
        os.path.join(
            MODEL_DIR,
            "y_test_cicids2017.npy"
        ),
        y_test
    )


    scaler_path = os.path.join(
        MODEL_DIR,
        "scaler_cicids2017.pkl"
    )


    with open(
        scaler_path,
        "wb"
    ) as f:

        pickle.dump(
            scaler,
            f
        )


    metadata = {

        "dataset":
            "CICIDS2017",

        "features":
            features,

        "num_features":
            len(features),

        "classes":
            CLASS_ORDER,

        "train_size":
            len(X_train),

        "test_size":
            len(X_test)
    }


    metadata_path = os.path.join(
        MODEL_DIR,
        "cicids2017_meta.json"
    )


    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )


    print(
        "\nArtifacts saved:"
    )

    print(
        "  models/X_train_cicids2017.npy"
    )

    print(
        "  models/X_test_cicids2017.npy"
    )

    print(
        "  models/y_train_cicids2017.npy"
    )

    print(
        "  models/y_test_cicids2017.npy"
    )

    print(
        "  models/scaler_cicids2017.pkl"
    )

    print(
        "  models/cicids2017_meta.json"
    )


def make_federated_split(
    X_train,
    y_train
):

    """
    Creates intentionally non-IID data.

    Hospital:
        mostly Normal + R2L

    Bank:
        mostly DoS + Probe

    Campus:
        remaining mixed data
    """

    rng = np.random.default_rng(
        42
    )


    primary_classes = {

        "hospital": [0, 3],

        "bank": [1, 2],

        "campus": [
            0,
            1,
            2,
            3,
            4
        ]
    }


    splits = {}


    # Hospital
    hospital_primary = np.where(
        np.isin(
            y_train,
            primary_classes["hospital"]
        )
    )[0]

    hospital_other = np.where(
        ~np.isin(
            y_train,
            primary_classes["hospital"]
        )
    )[0]


    hospital_primary = rng.choice(
        hospital_primary,
        int(
            len(hospital_primary) * 0.70
        ),
        replace=False
    )

    hospital_other = rng.choice(
        hospital_other,
        int(
            len(hospital_other) * 0.10
        ),
        replace=False
    )


    hospital_idx = np.concatenate(
        [
            hospital_primary,
            hospital_other
        ]
    )


    # Bank
    bank_primary = np.where(
        np.isin(
            y_train,
            primary_classes["bank"]
        )
    )[0]

    bank_other = np.where(
        ~np.isin(
            y_train,
            primary_classes["bank"]
        )
    )[0]


    bank_primary = rng.choice(
        bank_primary,
        int(
            len(bank_primary) * 0.70
        ),
        replace=False
    )

    bank_other = rng.choice(
        bank_other,
        int(
            len(bank_other) * 0.10
        ),
        replace=False
    )


    bank_idx = np.concatenate(
        [
            bank_primary,
            bank_other
        ]
    )


    used = set(
        hospital_idx.tolist()
    ) | set(
        bank_idx.tolist()
    )


    remaining = np.array(
        [
            index
            for index in range(
                len(y_train)
            )
            if index not in used
        ],
        dtype=int
    )


    splits["hospital"] = {
        "X": X_train[hospital_idx],
        "y": y_train[hospital_idx]
    }

    splits["bank"] = {
        "X": X_train[bank_idx],
        "y": y_train[bank_idx]
    }

    splits["campus"] = {
        "X": X_train[remaining],
        "y": y_train[remaining]
    }


    for name, data in splits.items():

        print(
            f"  {name.capitalize():8}: "
            f"{len(data['y']):,} samples | "
            f"Classes: "
            f"{np.unique(data['y']).tolist()}"
        )


    return splits


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "standard",
            "federated"
        ],
        default="standard"
    )

    args = parser.parse_args()


    print(
        "=" * 60
    )

    print(
        "FedShield - CICIDS2017 Preprocessing"
    )

    print(
        "=" * 60
    )


    if not os.path.exists(
        DATA_DIR
    ):

        os.makedirs(
            DATA_DIR,
            exist_ok=True
        )

        print(
            f"\nCreated directory:\n"
            f"{DATA_DIR}"
        )

        print(
            "\nPlace the CICIDS2017 CSV files "
            "inside this directory and run again."
        )

        raise SystemExit(0)


    print(
        "\n1. Loading dataset..."
    )

    df = load_cicids2017(
        DATA_DIR
    )


    print(
        "\n2. Preprocessing..."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        features
    ) = preprocess(
        df
    )


    print(
        "\n3. Creating federated split..."
    )

    if args.mode == "federated":

        splits = make_federated_split(
            X_train,
            y_train
        )


        for name, data in splits.items():

            np.save(
                os.path.join(
                    MODEL_DIR,
                    f"cicids2017_{name}_X.npy"
                ),
                data["X"]
            )

            np.save(
                os.path.join(
                    MODEL_DIR,
                    f"cicids2017_{name}_y.npy"
                ),
                data["y"]
            )


    print(
        "\n4. Saving artifacts..."
    )

    save_artifacts(
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        features
    )


    print(
        "\n✅ CICIDS2017 preprocessing complete"
    )

    print(
        f"Features: {len(features)}"
    )

    print(
        f"Train: {len(X_train):,}"
    )

    print(
        f"Test: {len(X_test):,}"
    )

    print(
        "\nNext:"
    )

    print(
        "python federated_noniid.py --dataset cicids2017"
    )