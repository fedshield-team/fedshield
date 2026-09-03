import os

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from model import MULTICLASS_FEATURE_NAMES


# ============================================================
# PATHS
# ============================================================

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
    DATA_DIR,
    exist_ok=True
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


# ============================================================
# NSL-KDD COLUMNS
# ============================================================

COLUMNS = [
    *MULTICLASS_FEATURE_NAMES,
    "label",
    "difficulty",
]


# ============================================================
# ATTACK → CLASS MAPPING
# ============================================================

ATTACK_MAP = {

    # Normal
    "normal": 0,

    # DoS
    "neptune": 1,
    "back": 1,
    "teardrop": 1,
    "pod": 1,
    "smurf": 1,
    "land": 1,
    "mailbomb": 1,
    "apache2": 1,

    # Probe
    "satan": 2,
    "ipsweep": 2,
    "portsweep": 2,
    "nmap": 2,
    "mscan": 2,

    # R2L
    "warezclient": 3,
    "warezmaster": 3,
    "imap": 3,
    "ftp_write": 3,
    "multihop": 3,
    "guess_passwd": 3,
    "phf": 3,
    "spy": 3,
    "sendmail": 3,

    # U2R
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


CATEGORICAL_COLUMNS = [
    "protocol_type",
    "service",
    "flag"
]


# ============================================================
# UNKNOWN CATEGORY HANDLING
# ============================================================

def fit_label_encoder(values):

    encoder = LabelEncoder()

    values = (
        pd.Series(values)
        .astype(str)
        .str.strip()
    )

    encoder.fit(values)

    return encoder


def transform_with_unknown(
    encoder,
    values
):

    classes = set(
        encoder.classes_
    )

    values = (
        pd.Series(values)
        .astype(str)
        .str.strip()
    )

    # LabelEncoder cannot handle unseen categories.
    # Use the first known category as fallback.
    fallback = encoder.classes_[0]

    values = values.map(
        lambda value:
        value if value in classes
        else fallback
    )

    return encoder.transform(
        values
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_multiclass(path):

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS
    )

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    labels = df["label"].map(
        ATTACK_MAP
    )

    valid = labels.notna()

    df = df.loc[
        valid
    ].copy()

    labels = labels.loc[
        valid
    ].astype(int)

    df.drop(
        columns=[
            "label",
            "difficulty"
        ],
        inplace=True
    )

    # --------------------------------------------------------
    # Split FIRST.
    # Encoders and scaler are fitted only on training data.
    # --------------------------------------------------------

    X_raw = df.copy()

    (
        X_train_df,
        X_test_df,
        y_train,
        y_test
    ) = train_test_split(
        X_raw,
        labels.values,
        test_size=0.20,
        random_state=42,
        stratify=labels.values
    )

    # --------------------------------------------------------
    # CATEGORICAL ENCODING
    # --------------------------------------------------------

    encoders = {}

    for column in CATEGORICAL_COLUMNS:

        encoder = fit_label_encoder(
            X_train_df[column]
        )

        # Pandas 3 may infer Arrow-backed string columns here.  Assigning
        # encoded integers into that dtype raises a TypeError, so explicitly
        # use an object column before replacing categories with numbers.
        X_train_df[column] = X_train_df[column].astype(object)
        X_test_df[column] = X_test_df[column].astype(object)

        X_train_df[column] = transform_with_unknown(
            encoder,
            X_train_df[column]
        )

        X_test_df[column] = transform_with_unknown(
            encoder,
            X_test_df[column]
        )

        encoders[column] = encoder

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    X_train = X_train_df.values.astype(
        np.float32
    )

    X_test = X_test_df.values.astype(
        np.float32
    )

    y_train = np.asarray(
        y_train,
        dtype=np.int64
    )

    y_test = np.asarray(
        y_test,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # STANDARD SCALING
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    ).astype(
        np.float32
    )

    X_test = scaler.transform(
        X_test
    ).astype(
        np.float32
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoders
    )


# ============================================================
# DISTRIBUTION
# ============================================================

def print_distribution(
    y,
    title
):

    print(
        f"\n{title}"
    )

    for index, name in enumerate(
        CLASS_NAMES
    ):

        count = int(
            np.sum(
                y == index
            )
        )

        print(
            f"  {name}: {count}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    input_path = os.path.join(
        DATA_DIR,
        "KDDTrain+.txt"
    )

    if not os.path.exists(
        input_path
    ):

        raise FileNotFoundError(
            f"Dataset not found:\n{input_path}"
        )

    print(
        "============================================================"
    )

    print(
        "FedShield — NSL-KDD Multi-Class Preprocessing"
    )

    print(
        "============================================================"
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

    print_distribution(
        y_train,
        "Training class distribution:"
    )

    print_distribution(
        y_test,
        "Test class distribution:"
    )

    # --------------------------------------------------------
    # SAVE ARRAYS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SAVE PREPROCESSORS
    # --------------------------------------------------------

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
        "\nSaved:"
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

    print(
        f"\nFeature count: "
        f"{X_train.shape[1]}"
    )