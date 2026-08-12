"""
FedShield — CICIDS2017 Dataset Support
Preprocesses the modern CICIDS2017 dataset for federated training.

CICIDS2017 vs NSL-KDD:
    - NSL-KDD: 1999 traffic, 41 features, 5 classes
    - CICIDS2017: 2017 traffic, 78 features, 15 attack types
    - More realistic — includes DoS, DDoS, PortScan, Brute Force, Web attacks

Download from: https://www.unb.ca/cic/datasets/ids-2017.html
Place CSV files in: data/cicids2017/

Usage:
    python cicids2017_preprocess.py
    python cicids2017_preprocess.py --mode federated  (creates 3-node split)
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import json

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, "data", "cicids2017")
MODEL_DIR = os.path.join(BASE, "models")

# ── CICIDS2017 label mapping → 5 unified classes ──────────────────────────────
LABEL_MAP = {
    "BENIGN":                    "Normal",
    "DoS Hulk":                  "DoS",
    "DoS GoldenEye":             "DoS",
    "DoS slowloris":             "DoS",
    "DoS Slowhttptest":          "DoS",
    "DDoS":                      "DoS",
    "Heartbleed":                "DoS",
    "PortScan":                  "Probe",
    "FTP-Patator":               "R2L",
    "SSH-Patator":               "R2L",
    "Web Attack – Brute Force":  "R2L",
    "Web Attack – XSS":          "R2L",
    "Web Attack – Sql Injection":"R2L",
    "Infiltration":              "U2R",
    "Bot":                       "U2R",
}

# Features matching NSL-KDD equivalents (network-level, no deep inspection)
SELECTED_FEATURES = [
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    "Total Length of Fwd Packets",
    " Total Length of Bwd Packets",
    " Fwd Packet Length Max",
    " Fwd Packet Length Min",
    " Fwd Packet Length Mean",
    " Bwd Packet Length Max",
    " Bwd Packet Length Min",
    " Flow Bytes/s",
    " Flow Packets/s",
    " Flow IAT Mean",
    " Flow IAT Std",
    " Fwd IAT Total",
    " Fwd IAT Mean",
    " Bwd IAT Total",
    " Bwd IAT Mean",
    " Fwd PSH Flags",
    " Bwd PSH Flags",
    " Fwd URG Flags",
    " Bwd URG Flags",
    " Fwd Header Length",
    " Bwd Header Length",
    " Fwd Packets/s",
    " Bwd Packets/s",
    " Min Packet Length",
    " Max Packet Length",
    " Packet Length Mean",
    " Packet Length Std",
    " Packet Length Variance",
    " FIN Flag Count",
    " SYN Flag Count",
    " RST Flag Count",
    " PSH Flag Count",
    " ACK Flag Count",
    " URG Flag Count",
    " CWE Flag Count",
    " ECE Flag Count",
    " Down/Up Ratio",
    " Average Packet Size",
]

CLASS_ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def load_cicids2017(data_dir: str) -> pd.DataFrame:
    """Load all CICIDS2017 CSV files from directory."""
    dfs = []
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}\n"
            f"Download from: https://www.unb.ca/cic/datasets/ids-2017.html"
        )

    for f in sorted(csv_files):
        path = os.path.join(data_dir, f)
        print(f"  Loading {f}...")
        try:
            df = pd.read_csv(path, encoding="latin-1", low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"  [WARN] Skipping {f}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total rows loaded: {len(combined):,}")
    return combined


def preprocess(df: pd.DataFrame) -> tuple:
    """Clean, map labels, select features, scale."""
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    label_col  = " Label" if " Label" in df.columns else "Label"

    # Map to 5-class labels
    df[label_col] = df[label_col].str.strip().map(LABEL_MAP)
    df = df.dropna(subset=[label_col])

    # Select available features
    available = [f.strip() for f in SELECTED_FEATURES if f.strip() in df.columns]
    print(f"  Features available: {len(available)}/{len(SELECTED_FEATURES)}")

    X = df[available].copy()
    y = df[label_col].copy()

    # Clean infinities and NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # Encode labels
    le = LabelEncoder()
    le.classes_ = np.array(CLASS_ORDER)
    y_enc = le.transform(y)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n  Class distribution:")
    for cls, count in zip(*np.unique(y, return_counts=True)):
        print(f"    {cls:10}: {count:8,}")

    return X_scaled, y_enc, scaler, le, available


def save_artifacts(X_train, X_test, y_train, y_test, scaler, features, mode="standard"):
    """Save preprocessed data and scaler."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    suffix = "_cicids2017"

    np.save(os.path.join(MODEL_DIR, f"X_train{suffix}.npy"), X_train)
    np.save(os.path.join(MODEL_DIR, f"X_test{suffix}.npy"),  X_test)
    np.save(os.path.join(MODEL_DIR, f"y_train{suffix}.npy"), y_train)
    np.save(os.path.join(MODEL_DIR, f"y_test{suffix}.npy"),  y_test)

    with open(os.path.join(MODEL_DIR, f"scaler{suffix}.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "dataset":      "CICIDS2017",
        "features":     features,
        "num_features": len(features),
        "classes":      CLASS_ORDER,
        "train_size":   len(X_train),
        "test_size":    len(X_test),
    }
    with open(os.path.join(MODEL_DIR, f"cicids2017_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved to models/ with suffix {suffix}")


def make_federated_split(X_train, y_train):
    """
    Create non-IID federated split across 3 nodes.
    Hospital: Normal + R2L  (classes 0, 3)
    Bank:     DoS + Probe   (classes 1, 2)
    Campus:   Mixed         (all classes)
    """
    nodes = {
        "hospital": {"primary": [0, 3], "ratio": 0.7},
        "bank":     {"primary": [1, 2], "ratio": 0.7},
        "campus":   {"primary": [0, 1, 2, 3, 4], "ratio": 0.3},
    }

    splits = {}
    for name, cfg in nodes.items():
        primary_mask = np.isin(y_train, cfg["primary"])
        other_mask   = ~primary_mask

        primary_idx = np.where(primary_mask)[0]
        other_idx   = np.where(other_mask)[0]

        n_primary = int(len(primary_idx) * cfg["ratio"])
        n_other   = int(len(other_idx) * 0.1)

        chosen = np.concatenate([
            np.random.choice(primary_idx, min(n_primary, len(primary_idx)), replace=False),
            np.random.choice(other_idx,   min(n_other,   len(other_idx)),   replace=False),
        ])
        np.random.shuffle(chosen)

        splits[name] = {"X": X_train[chosen], "y": y_train[chosen]}
        print(f"  {name:10}: {len(chosen):6,} samples | classes: {np.unique(y_train[chosen])}")

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["standard", "federated"], default="standard")
    args = parser.parse_args()

    print("=" * 60)
    print("FedShield — CICIDS2017 Preprocessing")
    print("=" * 60)

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"\n⚠️  Data directory created: {DATA_DIR}")
        print("   Download CICIDS2017 CSV files from:")
        print("   https://www.unb.ca/cic/datasets/ids-2017.html")
        print("   Place all CSV files in:", DATA_DIR)
        print("\n   Files needed:")
        print("   - Monday-WorkingHours.pcap_ISCX.csv")
        print("   - Tuesday-WorkingHours.pcap_ISCX.csv")
        print("   - Wednesday-workingHours.pcap_ISCX.csv")
        print("   - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
        print("   - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
        print("   - Friday-WorkingHours-Morning.pcap_ISCX.csv")
        print("   - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
        print("   - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
        exit(0)

    print("\n1. Loading CSV files...")
    df = load_cicids2017(DATA_DIR)

    print("\n2. Preprocessing...")
    X, y, scaler, le, features = preprocess(df)

    print("\n3. Train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    if args.mode == "federated":
        print("\n4. Creating non-IID federated split...")
        splits = make_federated_split(X_train, y_train)
        for name, data in splits.items():
            np.save(os.path.join(MODEL_DIR, f"cicids2017_{name}_X.npy"), data["X"])
            np.save(os.path.join(MODEL_DIR, f"cicids2017_{name}_y.npy"), data["y"])
        print("  Saved federated splits to models/")

    print("\n5. Saving artifacts...")
    save_artifacts(X_train, X_test, y_train, y_test, scaler, features)

    print("\n✅ CICIDS2017 preprocessing complete")
    print(f"   Features: {len(features)}")
    print(f"   Train:    {len(X_train):,}")
    print(f"   Test:     {len(X_test):,}")
    print("\nNext step:")
    print("  python federated_noniid.py --dataset cicids2017")
    print("=" * 60)