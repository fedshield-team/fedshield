"""
FedShield — Live packet capture, detection, explainability and response.

Updated version:
- Packet/flow-aware feature extraction
- Short-term connection statistics
- Safer ML attack confirmation
- Independent port-scan detection
- SHAP explanations
- SQLite logging
- JSON live snapshot
- Online retraining
- Hot model reload
- Windows Firewall blocking for confirmed port scans only
"""

import json
import os
import subprocess
import threading
import time
import uuid
import sqlite3
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import joblib
import shap
import torch
import torch.nn as nn

from scapy.all import (
    sniff,
    IP,
    TCP,
    UDP,
)

from llm_incident_report import (
    generate_incident_report,
    init_incident_reports_table,
)

from online_retrain import (
    init_retrain_buffer,
    log_retrain_sample,
    should_retrain,
    run_incremental_retrain,
    VERSION_PATH,
)


# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE,
    "models",
    "federated_noniid_model.pth",
)

SCALER_PATH = os.path.join(
    BASE,
    "models",
    "scaler_multiclass.pkl",
)

DB_PATH = os.path.join(
    BASE,
    "models",
    "fedshield_logs.db",
)

LIVE_LOG_PATH = os.path.join(
    BASE,
    "models",
    "live_log.json",
)

TRAIN_BG_PATH = os.path.join(
    BASE,
    "data",
    "X_train_mc.npy",
)


# ============================================================
# CLASS / FEATURE DEFINITIONS
# ============================================================

CLASS_NAMES = [
    "Normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R",
]


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
    "dst_host_srv_rerror_rate",
]


PROTO_MAP = {
    "icmp": 0,
    "tcp": 1,
    "udp": 2,
}


# Common NSL-KDD service approximation.
# If your training pipeline has a dedicated encoder,
# the code below will try to use it.
SERVICE_MAP = {
    20: 0,
    21: 1,
    22: 2,
    23: 3,
    25: 4,
    53: 5,
    80: 6,
    110: 7,
    111: 8,
    119: 9,
    135: 10,
    139: 11,
    143: 12,
    443: 13,
    445: 14,
    993: 15,
    995: 16,
    1433: 17,
    3306: 18,
    3389: 19,
    8080: 20,
}


# TCP flag approximation.
FLAG_MAP = {
    "S": 0,
    "SA": 1,
    "A": 2,
    "FA": 3,
    "F": 4,
    "RA": 5,
    "R": 6,
    "PA": 7,
    "P": 8,
}


# ============================================================
# MODEL
# ============================================================

class MultiClassIDS(nn.Module):

    def __init__(
        self,
        input_dim=41,
        num_classes=5,
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                256,
            ),

            nn.BatchNorm1d(
                256,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3,
            ),

            nn.Linear(
                256,
                128,
            ),

            nn.BatchNorm1d(
                128,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2,
            ),

            nn.Linear(
                128,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                num_classes,
            ),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# DIRECTORY SETUP
# ============================================================

os.makedirs(
    os.path.join(BASE, "models"),
    exist_ok=True,
)


# ============================================================
# DATABASE
# ============================================================

def init_db():

    conn = sqlite3.connect(
        DB_PATH,
        timeout=10,
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            src TEXT,
            dst TEXT,
            proto TEXT,
            prediction TEXT,
            confidence REAL,
            tag TEXT,
            blocked INTEGER DEFAULT 0,
            incident_id TEXT
        )
        """
    )

    columns = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(detections)"
        )
    }

    if "blocked" not in columns:

        conn.execute(
            """
            ALTER TABLE detections
            ADD COLUMN blocked INTEGER DEFAULT 0
            """
        )

    if "incident_id" not in columns:

        conn.execute(
            """
            ALTER TABLE detections
            ADD COLUMN incident_id TEXT
            """
        )

    conn.commit()

    return conn


def db_insert_detection(entry):

    with sqlite3.connect(
        DB_PATH,
        timeout=10,
    ) as conn:

        cursor = conn.execute(
            """
            INSERT INTO detections
            (
                timestamp,
                src,
                dst,
                proto,
                prediction,
                confidence,
                tag,
                incident_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["timestamp"],
                entry["src"],
                entry["dst"],
                entry["proto"],
                entry["prediction"],
                entry["confidence"],
                entry["tag"],
                entry["incident_id"],
            ),
        )

        conn.commit()

        return cursor.lastrowid


def mark_blocked(row_id):

    with sqlite3.connect(
        DB_PATH,
        timeout=10,
    ) as conn:

        conn.execute(
            """
            UPDATE detections
            SET blocked = 1
            WHERE id = ?
            """,
            (row_id,),
        )

        conn.commit()


db_conn = init_db()
db_conn.close()

init_incident_reports_table()
init_retrain_buffer()


# ============================================================
# LOAD MODEL
# ============================================================

model = MultiClassIDS()

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )
)

model.eval()


# ============================================================
# LOAD SCALER
# ============================================================

if not os.path.exists(SCALER_PATH):

    raise FileNotFoundError(
        f"Scaler not found: {SCALER_PATH}"
    )


scaler = joblib.load(
    SCALER_PATH
)

print("Model + scaler loaded.")


# ============================================================
# MODEL VERSION
# ============================================================

current_model_version = None

if os.path.exists(VERSION_PATH):

    try:

        with open(
            VERSION_PATH,
            encoding="utf-8",
        ) as f:

            current_model_version = (
                json.load(f).get("version")
            )

    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
    ):

        current_model_version = None


# ============================================================
# SHAP
# ============================================================

try:

    background = np.load(
        TRAIN_BG_PATH
    )[:100]

    shap_explainer = shap.DeepExplainer(
        model,
        torch.as_tensor(
            background,
            dtype=torch.float32,
        ),
    )

    print(
        "SHAP DeepExplainer initialised."
    )

except Exception as e:

    shap_explainer = None

    print(
        f"[WARN] SHAP disabled: {e}"
    )


def get_top_shap_features(
    x_tensor,
    pred_class,
    top_n=3,
):

    if shap_explainer is None:
        return []

    try:

        values = shap_explainer.shap_values(
            x_tensor
        )

        if isinstance(values, list):

            class_values = np.asarray(
                values[pred_class]
            )[0]

        else:

            arr = np.asarray(values)

            if (
                arr.ndim == 3
                and arr.shape[1] == len(FEATURE_NAMES)
            ):

                class_values = arr[
                    0,
                    :,
                    pred_class,
                ]

            elif (
                arr.ndim == 3
                and arr.shape[2] == len(FEATURE_NAMES)
            ):

                class_values = arr[
                    pred_class,
                    0,
                    :,
                ]

            elif arr.ndim == 2:

                class_values = arr[0]

            else:

                class_values = arr.reshape(-1)

        if len(class_values) != len(FEATURE_NAMES):
            return []

        indices = np.argsort(
            np.abs(class_values)
        )[::-1][:top_n]

        return [
            (
                FEATURE_NAMES[i],
                float(class_values[i]),
            )
            for i in indices
        ]

    except Exception as e:

        print(
            f"[WARN] SHAP computation failed: {e}"
        )

        return []


# ============================================================
# LIVE STATE
# ============================================================

reported_incidents = set()

log = []

blocked_ips = set()

alerted_pairs = set()

retrain_lock = threading.Lock()

retrain_in_progress = False


# ------------------------------------------------------------
# General short-term windows
# ------------------------------------------------------------

WINDOW_SECONDS = 10

recent_packets = deque(
    maxlen=5000
)


# ------------------------------------------------------------
# Flow state
# ------------------------------------------------------------

flow_state = defaultdict(
    lambda: {
        "first_seen": None,
        "last_seen": None,
        "packets": 0,
        "src_bytes": 0,
        "dst_bytes": 0,
        "syn": 0,
        "ack": 0,
        "rst": 0,
        "fin": 0,
        "errors": 0,
        "services": set(),
    }
)


# ------------------------------------------------------------
# Host state
# ------------------------------------------------------------

host_stats = defaultdict(
    lambda: {
        "timestamps": deque(maxlen=500),
        "dsts": deque(maxlen=500),
        "ports": deque(maxlen=500),
        "services": deque(maxlen=500),
        "errors": 0,
        "syn": 0,
        "packets": 0,
    }
)


# ============================================================
# DETECTION THRESHOLDS
# ============================================================

# IMPORTANT:
# ML predictions alone should not automatically block hosts.

ML_ATTACK_THRESHOLD = 0.92

ML_CONFIRMATIONS_REQUIRED = 3

ML_CONFIRMATION_WINDOW = 10

PROBE_CONFIDENCE_THRESHOLD = 0.85

SCAN_PORT_THRESHOLD = 8

SCAN_WINDOW_SECONDS = 3


# ============================================================
# ML CONFIRMATION STATE
# ============================================================

attack_evidence = defaultdict(
    lambda: deque()
)


def register_attack_evidence(
    src,
    dst,
    label,
    confidence,
):

    now = time.monotonic()

    key = (
        src,
        dst,
        label,
    )

    evidence = attack_evidence[key]

    while evidence:

        if (
            now - evidence[0]
            > ML_CONFIRMATION_WINDOW
        ):

            evidence.popleft()

        else:

            break

    if confidence >= ML_ATTACK_THRESHOLD:

        evidence.append(now)

    return len(evidence)


# ============================================================
# FIREWALL
# ============================================================

def block_ip(ip):

    """
    Block an IP using Windows Firewall.

    Only used for independently confirmed
    port scans.

    Requires Administrator privileges.
    """

    import ipaddress

    try:

        ipaddress.ip_address(ip)

    except ValueError:

        print(
            f"[WARN] Invalid IP: {ip}"
        )

        return False

    if ip in blocked_ips:
        return False

    rule_name = (
        "FedShield_Block_"
        + ip.replace(":", "_")
        .replace(".", "_")
    )

    cmd = [
        "netsh",
        "advfirewall",
        "firewall",
        "add",
        "rule",
        f"name={rule_name}",
        "dir=in",
        "action=block",
        f"remoteip={ip}",
    ]

    try:

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=False,
            timeout=10,
        )

        if result.returncode == 0:

            blocked_ips.add(ip)

            return True

        print(
            f"[WARN] Could not block {ip}: "
            f"{result.stderr.strip()}"
        )

    except (
        OSError,
        subprocess.SubprocessError,
    ) as e:

        print(
            f"[WARN] Block failed for {ip}: {e}"
        )

    return False


# ============================================================
# PORT SCAN DETECTION
# ============================================================

port_scan_tracker = defaultdict(
    lambda: {
        "ports": set(),
        "first_seen": None,
    }
)


def check_port_scan(
    src,
    dst,
    dport,
    proto,
):

    if (
        proto != "tcp"
        or dport is None
    ):

        return False

    key = (
        src,
        dst,
    )

    now = time.monotonic()

    entry = port_scan_tracker[key]

    if (
        entry["first_seen"] is None
        or
        now - entry["first_seen"]
        > SCAN_WINDOW_SECONDS
    ):

        entry["first_seen"] = now
        entry["ports"] = set()

    entry["ports"].add(
        int(dport)
    )

    if (
        len(entry["ports"])
        >= SCAN_PORT_THRESHOLD
        and
        key not in alerted_pairs
    ):

        alerted_pairs.add(key)

        return True

    return False


# ============================================================
# PACKET HELPERS
# ============================================================

def get_protocol(pkt):

    if TCP in pkt:
        return "tcp"

    if UDP in pkt:
        return "udp"

    return "icmp"


def get_destination_port(pkt):

    if TCP in pkt:
        return int(pkt[TCP].dport)

    if UDP in pkt:
        return int(pkt[UDP].dport)

    return None


def get_source_port(pkt):

    if TCP in pkt:
        return int(pkt[TCP].sport)

    if UDP in pkt:
        return int(pkt[UDP].sport)

    return None


def get_tcp_flags(pkt):

    if TCP not in pkt:
        return ""

    try:
        return str(pkt[TCP].flags)
    except Exception:
        return ""


def get_service_code(dport):

    if dport is None:
        return 0

    if dport in SERVICE_MAP:
        return SERVICE_MAP[dport]

    # Common web / application services
    if dport in (8000, 8008, 8080, 8081, 8443):
        return 20

    # Keep unknown services separated from known services.
    return 21


def get_flag_code(flags):

    if not flags:
        return 0

    if flags in FLAG_MAP:
        return FLAG_MAP[flags]

    # Normalize common Scapy representations.
    if "S" in flags and "A" in flags:
        return FLAG_MAP["SA"]

    if "S" in flags:
        return FLAG_MAP["S"]

    if "R" in flags and "A" in flags:
        return FLAG_MAP["RA"]

    if "R" in flags:
        return FLAG_MAP["R"]

    if "F" in flags and "A" in flags:
        return FLAG_MAP["FA"]

    if "F" in flags:
        return FLAG_MAP["F"]

    if "P" in flags and "A" in flags:
        return FLAG_MAP["PA"]

    if "P" in flags:
        return FLAG_MAP["P"]

    if "A" in flags:
        return FLAG_MAP["A"]

    return 0


# ============================================================
# CLEAN OLD STATE
# ============================================================

def cleanup_state():

    now = time.monotonic()

    # Packet history
    while recent_packets:

        if (
            now - recent_packets[0]["time"]
            > WINDOW_SECONDS
        ):

            recent_packets.popleft()

        else:

            break

    # Host history
    for host, stats in list(
        host_stats.items()
    ):

        timestamps = stats["timestamps"]

        while timestamps:

            if (
                now - timestamps[0]
                > WINDOW_SECONDS
            ):

                timestamps.popleft()

            else:

                break

        if not timestamps:

            host_stats.pop(
                host,
                None,
            )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(pkt):

    """
    Convert a live packet into a 41-feature
    NSL-KDD-style numerical representation.

    IMPORTANT:
    A live packet cannot reproduce every NSL-KDD
    feature exactly because NSL-KDD is connection/
    flow based.

    Therefore this function uses short-term
    connection and host statistics instead of
    filling almost everything with zeros.
    """

    if IP not in pkt:

        return None, None

    cleanup_state()

    now = time.monotonic()

    proto = get_protocol(pkt)

    src = pkt[IP].src
    dst = pkt[IP].dst

    dport = get_destination_port(pkt)
    sport = get_source_port(pkt)

    packet_length = int(
        len(pkt)
    )

    flags = get_tcp_flags(pkt)

    service_code = get_service_code(
        dport
    )

    flag_code = get_flag_code(
        flags
    )

    flow_key = (
        src,
        dst,
        sport,
        dport,
        proto,
    )

    flow = flow_state[
        flow_key
    ]

    if flow["first_seen"] is None:

        flow["first_seen"] = now

    flow["last_seen"] = now

    flow["packets"] += 1

    flow["src_bytes"] += packet_length

    if TCP in pkt:

        if "S" in flags:
            flow["syn"] += 1

        if "A" in flags:
            flow["ack"] += 1

        if "R" in flags:
            flow["rst"] += 1

        if "F" in flags:
            flow["fin"] += 1

    host = host_stats[src]

    host["timestamps"].append(
        now
    )

    host["dsts"].append(
        dst
    )

    if dport is not None:

        host["ports"].append(
            dport
        )

        host["services"].append(
            service_code
        )

    host["packets"] += 1

    # A SYN without ACK is a useful signal,
    # but it is NOT automatically an error.
    if (
        TCP in pkt
        and "S" in flags
        and "A" not in flags
    ):

        host["syn"] += 1

    if (
        TCP in pkt
        and "R" in flags
    ):

        host["errors"] += 1

    # --------------------------------------------------------
    # Short-term host statistics
    # --------------------------------------------------------

    count = len(
        host["timestamps"]
    )

    unique_dsts = len(
        set(host["dsts"])
    )

    unique_ports = len(
        set(host["ports"])
    )

    unique_services = len(
        set(host["services"])
    )

    syn_count = host["syn"]

    error_count = host["errors"]

    serror_rate = (
        syn_count
        /
        max(count, 1)
    )

    rerror_rate = (
        error_count
        /
        max(count, 1)
    )

    same_dst_count = sum(
        1
        for d in host["dsts"]
        if d == dst
    )

    same_service_count = sum(
        1
        for s in host["services"]
        if s == service_code
    )

    same_dst_rate = (
        same_dst_count
        /
        max(count, 1)
    )

    same_service_rate = (
        same_service_count
        /
        max(len(host["services"]), 1)
    )

    diff_service_rate = (
        1.0 - same_service_rate
    )

    # --------------------------------------------------------
    # Destination-host statistics
    # --------------------------------------------------------

    destination_packets = 0
    destination_services = 0
    destination_same_source = 0
    destination_errors = 0

    for item in recent_packets:

        if item["dst"] != dst:
            continue

        destination_packets += 1

        if item["service"] == service_code:

            destination_services += 1

        if item["src"] == src:

            destination_same_source += 1

        if item["error"]:

            destination_errors += 1

    dst_host_count = destination_packets

    dst_host_srv_count = (
        destination_services
    )

    dst_host_same_srv_rate = (
        destination_services
        /
        max(destination_packets, 1)
    )

    dst_host_diff_srv_rate = (
        1.0
        -
        dst_host_same_srv_rate
    )

    dst_host_same_src_port_rate = (
        destination_same_source
        /
        max(destination_packets, 1)
    )

    dst_host_srv_diff_host_rate = (
        1.0
        -
        dst_host_same_src_port_rate
    )

    dst_host_serror_rate = (
        destination_errors
        /
        max(destination_packets, 1)
    )

    dst_host_srv_serror_rate = (
        dst_host_serror_rate
    )

    dst_host_rerror_rate = (
        destination_errors
        /
        max(destination_packets, 1)
    )

    dst_host_srv_rerror_rate = (
        dst_host_rerror_rate
    )

    # --------------------------------------------------------
    # Create all 41 features
    # --------------------------------------------------------

    features = np.zeros(
        41,
        dtype=np.float32,
    )

    # 0 duration
    features[0] = min(
        now - flow["first_seen"],
        3600.0,
    )

    # 1 protocol
    features[1] = PROTO_MAP.get(
        proto,
        0,
    )

    # 2 service
    features[2] = service_code

    # 3 flag
    features[3] = flag_code

    # 4 source bytes
    features[4] = min(
        flow["src_bytes"],
        10_000_000,
    )

    # 5 destination bytes
    features[5] = min(
        packet_length,
        10_000_000,
    )

    # 6 land
    features[6] = float(
        src == dst
    )

    # 7 wrong fragment
    features[7] = float(
        getattr(
            pkt[IP],
            "frag",
            0,
        ) > 0
    )

    # 8 urgent
    features[8] = float(
        TCP in pkt
        and
        getattr(
            pkt[TCP],
            "urgptr",
            0,
        ) > 0
    )

    # 9 hot
    features[9] = min(
        unique_services,
        255,
    )

    # 10 failed logins
    features[10] = 0

    # 11 logged in
    # For live traffic this is approximated by
    # established TCP traffic.
    features[11] = float(
        TCP in pkt
        and
        "A" in flags
    )

    # 12 compromised
    features[12] = 0

    # 13 root shell
    features[13] = 0

    # 14 su attempted
    features[14] = 0

    # 15 num root
    features[15] = 0

    # 16 file creations
    features[16] = 0

    # 17 shells
    features[17] = 0

    # 18 access files
    features[18] = 0

    # 19 outbound commands
    features[19] = 0

    # 20 host login
    features[20] = 0

    # 21 guest login
    features[21] = 0

    # 22 count
    features[22] = min(
        count,
        255,
    )

    # 23 srv_count
    features[23] = min(
        same_service_count,
        255,
    )

    # 24 serror_rate
    features[24] = serror_rate

    # 25 srv_serror_rate
    features[25] = serror_rate

    # 26 rerror_rate
    features[26] = rerror_rate

    # 27 srv_rerror_rate
    features[27] = rerror_rate

    # 28 same_srv_rate
    features[28] = same_service_rate

    # 29 diff_srv_rate
    features[29] = diff_service_rate

    # 30 srv_diff_host_rate
    features[30] = (
        1.0
        -
        (
            same_dst_count
            /
            max(count, 1)
        )
    )

    # 31 dst_host_count
    features[31] = min(
        dst_host_count,
        255,
    )

    # 32 dst_host_srv_count
    features[32] = min(
        dst_host_srv_count,
        255,
    )

    # 33 dst_host_same_srv_rate
    features[33] = dst_host_same_srv_rate

    # 34 dst_host_diff_srv_rate
    features[34] = dst_host_diff_srv_rate

    # 35 dst_host_same_src_port_rate
    features[35] = dst_host_same_src_port_rate

    # 36 dst_host_srv_diff_host_rate
    features[36] = dst_host_srv_diff_host_rate

    # 37 dst_host_serror_rate
    features[37] = dst_host_serror_rate

    # 38 dst_host_srv_serror_rate
    features[38] = dst_host_srv_serror_rate

    # 39 dst_host_rerror_rate
    features[39] = dst_host_rerror_rate

    # 40 dst_host_srv_rerror_rate
    features[40] = dst_host_srv_rerror_rate

    # --------------------------------------------------------
    # Save packet in recent history
    # --------------------------------------------------------

    recent_packets.append(
        {
            "time": now,
            "src": src,
            "dst": dst,
            "proto": proto,
            "sport": sport,
            "dport": dport,
            "service": service_code,
            "error": bool(
                TCP in pkt
                and "R" in flags
            ),
        }
    )

    meta = {
        "src": src,
        "dst": dst,
        "proto": proto,
        "sport": sport,
        "dport": dport,
        "service": service_code,
        "flag": flags,
        "count": count,
        "unique_ports": unique_ports,
        "unique_dsts": unique_dsts,
        "unique_services": unique_services,
        "syn_count": syn_count,
        "error_count": error_count,
        "flow_packets": flow["packets"],
        "flow_duration": (
            now - flow["first_seen"]
        ),
    }

    return features, meta


# ============================================================
# INCIDENT REPORT
# ============================================================

def maybe_generate_incident_report(
    meta,
    label,
    confidence,
    x_tensor,
    pred_class,
    force=False,
):

    key = (
        meta["src"],
        meta["dst"],
        label,
    )

    if (
        not force
        and
        key in reported_incidents
    ):

        return None

    reported_incidents.add(key)

    incident_id = str(
        uuid.uuid4()
    )

    flow_stats = {

        "src_ip":
            meta["src"],

        "dst_ip":
            meta["dst"],

        "src_port":
            meta["sport"]
            if meta["sport"] is not None
            else "N/A",

        "dst_port":
            meta["dport"]
            if meta["dport"] is not None
            else "N/A",

        "protocol":
            meta["proto"],

        "packet_count_this_host":
            meta["count"],

        "flow_packets":
            meta["flow_packets"],

        "flow_duration":
            round(
                meta["flow_duration"],
                3,
            ),

        "unique_ports":
            meta["unique_ports"],

        "unique_destinations":
            meta["unique_dsts"],

        "syn_count":
            meta["syn_count"],

        "error_count":
            meta["error_count"],
    }

    shap_features = (
        get_top_shap_features(
            x_tensor,
            pred_class,
        )
    )

    report = generate_incident_report(
        incident_id,
        flow_stats,
        label,
        confidence,
        shap_features,
    )

    print(
        f"\nINCIDENT REPORT "
        f"[{incident_id[:8]}]:\n"
        f"{report}\n"
    )

    return incident_id


# ============================================================
# ONLINE RETRAINING
# ============================================================

def maybe_trigger_retrain():

    global retrain_in_progress

    with retrain_lock:

        if retrain_in_progress:
            return

        retrain_in_progress = True

    def worker():

        global retrain_in_progress

        try:

            if should_retrain():

                result = (
                    run_incremental_retrain()
                )

                print(
                    "[online_retrain] "
                    f"{result.get(
                        'reason',
                        result.get(
                            'error',
                            'done',
                        ),
                    )}"
                )

        except Exception as e:

            print(
                f"[WARN] Online retraining failed: {e}"
            )

        finally:

            with retrain_lock:

                retrain_in_progress = False

    threading.Thread(
        target=worker,
        daemon=True,
    ).start()


# ============================================================
# HOT RELOAD
# ============================================================

def maybe_hot_reload_model():

    global current_model_version

    if not os.path.exists(
        VERSION_PATH
    ):

        return

    try:

        with open(
            VERSION_PATH,
            encoding="utf-8",
        ) as f:

            info = json.load(f)

        version = info.get(
            "version"
        )

        if (
            version
            and
            version != current_model_version
        ):

            state = torch.load(
                MODEL_PATH,
                map_location="cpu",
                weights_only=True,
            )

            model.load_state_dict(
                state
            )

            model.eval()

            current_model_version = version

            print(
                "[online_retrain] "
                f"Hot-reloaded model "
                f"version {version}."
            )

    except (
        OSError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ) as e:

        print(
            f"[WARN] Hot reload failed: {e}"
        )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_packet(pkt):

    raw_features, meta = (
        extract_features(pkt)
    )

    if raw_features is None:
        return

    # --------------------------------------------------------
    # SCALE FEATURES
    # --------------------------------------------------------

    try:

        scaled = (
            scaler.transform(
                raw_features.reshape(
                    1,
                    -1,
                )
            )
            .astype(
                np.float32
            )
        )

    except Exception as e:

        print(
            f"[WARN] Scaling failed: {e}"
        )

        return

    x = torch.as_tensor(
        scaled,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    try:

        with torch.no_grad():

            logits = model(x)

            probs = torch.softmax(
                logits,
                dim=1,
            )

            pred_class = int(
                torch.argmax(
                    probs,
                    dim=1,
                ).item()
            )

            confidence = float(
                probs[
                    0,
                    pred_class,
                ].item()
            )

    except Exception as e:

        print(
            f"[WARN] Model inference failed: {e}"
        )

        return

    label = CLASS_NAMES[
        pred_class
    ]

    # --------------------------------------------------------
    # INDEPENDENT PORT SCAN RULE
    # --------------------------------------------------------

    scan_detected = (
        check_port_scan(
            meta["src"],
            meta["dst"],
            meta["dport"],
            meta["proto"],
        )
    )

    incident_id = None

    # ========================================================
    # PORT SCAN
    # ========================================================

    if scan_detected:

        probe_class = CLASS_NAMES.index(
            "Probe"
        )

        incident_id = (
            maybe_generate_incident_report(
                meta,
                "Probe",
                max(
                    confidence,
                    PROBE_CONFIDENCE_THRESHOLD,
                ),
                x,
                probe_class,
                force=True,
            )
        )

    # ========================================================
    # ML ATTACK
    # ========================================================

    elif (
        pred_class != 0
        and
        confidence >= ML_ATTACK_THRESHOLD
    ):

        evidence_count = (
            register_attack_evidence(
                meta["src"],
                meta["dst"],
                label,
                confidence,
            )
        )

        # Do NOT call it a confirmed incident
        # from one packet alone.
        if (
            evidence_count
            >= ML_CONFIRMATIONS_REQUIRED
        ):

            incident_id = (
                maybe_generate_incident_report(
                    meta,
                    label,
                    confidence,
                    x,
                    pred_class,
                )
            )

    else:

        # A prediction below threshold is treated
        # as suspicious/uncertain rather than
        # immediately becoming an attack.
        if (
            pred_class != 0
            and
            confidence >= 0.70
        ):

            print(
                f"[SUSPICIOUS] "
                f"{meta['src']} -> "
                f"{meta['dst']} => "
                f"{label} "
                f"({confidence:.2%})"
            )

    # ========================================================
    # FINAL TAG
    # ========================================================

    confirmed_ml_attack = (
        pred_class != 0
        and
        confidence >= ML_ATTACK_THRESHOLD
        and
        incident_id is not None
    )

    if scan_detected:

        tag = "ATTACK"

    elif confirmed_ml_attack:

        tag = "ATTACK"

    elif (
        pred_class != 0
        and
        confidence >= 0.70
    ):

        tag = "SUSPICIOUS"

    else:

        tag = "normal"

    # ========================================================
    # DATABASE ENTRY
    # ========================================================

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    entry = {

        "timestamp":
            timestamp,

        "time":
            timestamp,

        "src":
            meta["src"],

        "dst":
            meta["dst"],

        "proto":
            meta["proto"],

        "prediction":
            label,

        "confidence":
            round(
                confidence,
                3,
            ),

        "tag":
            tag,

        "incident_id":
            incident_id,

        "dst_port":
            meta["dport"],

        "src_port":
            meta["sport"],

        "flow_packets":
            meta["flow_packets"],

        "flow_duration":
            round(
                meta["flow_duration"],
                3,
            ),

        "unique_ports":
            meta["unique_ports"],
    }

    try:

        row_id = (
            db_insert_detection(
                entry
            )
        )

    except Exception as e:

        row_id = None

        print(
            f"[WARN] Database insert failed: {e}"
        )

    log.append(
        entry
    )

    # ========================================================
    # PORT SCAN RESPONSE
    # ========================================================

    if scan_detected:

        print(
            f"\nPORT SCAN DETECTED: "
            f"{meta['src']} -> "
            f"{meta['dst']} "
            f"("
            f"{SCAN_PORT_THRESHOLD}+ "
            f"ports/"
            f"{SCAN_WINDOW_SECONDS}s"
            f")"
        )

        # Only the independent rule-based
        # port scan can automatically trigger
        # firewall blocking.

        if block_ip(
            meta["src"]
        ):

            if row_id is not None:

                mark_blocked(
                    row_id
                )

            print(
                f"AUTO-BLOCKED: "
                f"{meta['src']}"
            )

        # Store rule-confirmed Probe sample.
        try:

            log_retrain_sample(
                scaled[0],
                CLASS_NAMES.index(
                    "Probe"
                ),
                "rule_confirmed_probe",
            )

        except Exception as e:

            print(
                f"[WARN] Could not store "
                f"Probe retraining sample: {e}"
            )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    elif confirmed_ml_attack:

        print(
            f"ATTACK: "
            f"{meta['src']} -> "
            f"{meta['dst']} => "
            f"{label} "
            f"({confidence:.2%}) "
            f"[confirmed]"
        )

    elif (
        pred_class != 0
        and
        confidence >= 0.70
    ):

        print(
            f"suspicious: "
            f"{meta['src']} -> "
            f"{meta['dst']} => "
            f"{label} "
            f"({confidence:.2%})"
        )

    else:

        print(
            f"normal: "
            f"{meta['src']} -> "
            f"{meta['dst']} "
            f"({confidence:.2%})"
        )

    # ========================================================
    # JSON SNAPSHOT
    # ========================================================

    if len(log) % 5 == 0:

        try:

            with open(
                LIVE_LOG_PATH,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    log[-200:],
                    f,
                    indent=2,
                )

        except OSError as e:

            print(
                f"[WARN] Could not write "
                f"live log: {e}"
            )

    # ========================================================
    # RETRAINING / HOT RELOAD
    # ========================================================

    if len(log) % 25 == 0:

        maybe_hot_reload_model()

        maybe_trigger_retrain()


# ============================================================
# START
# ============================================================

print(
    "\n============================================================"
)

print(
    "          FedShield Live Capture Started"
)

print(
    "============================================================"
)

print(
    f"Port scan detection: "
    f"{SCAN_PORT_THRESHOLD}+ distinct TCP ports "
    f"in {SCAN_WINDOW_SECONDS}s"
)

print(
    f"ML attack threshold: "
    f"{ML_ATTACK_THRESHOLD:.0%}"
)

print(
    f"ML confirmation: "
    f"{ML_CONFIRMATIONS_REQUIRED} high-confidence "
    f"events within {ML_CONFIRMATION_WINDOW}s"
)

print(
    "Auto-block: ENABLED for confirmed port scans only"
)

print(
    "ML predictions: LOG + EXPLAIN, no direct firewall block"
)

print(
    "SHAP: ENABLED when background data is available"
)

print(
    "Online retraining: ENABLED"
)

print(
    "Hot model reload: ENABLED"
)

print(
    "Press Ctrl+C to stop.\n"
)


# ============================================================
# PACKET CAPTURE
# ============================================================

try:

    sniff(
        prn=classify_packet,
        store=False,
        count=0,
    )

except KeyboardInterrupt:

    print(
        "\nCapture stopped."
    )

except Exception as e:

    print(
        f"\n[ERROR] Packet capture failed: {e}"
    )

finally:

    try:

        with open(
            LIVE_LOG_PATH,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                log[-200:],
                f,
                indent=2,
            )

    except OSError as e:

        print(
            f"[WARN] Final log write failed: {e}"
        )

    print(
        f"\nSession packets: "
        f"{len(log)} | "
        f"Blocked IPs: "
        f"{len(blocked_ips)}"
    )

    print(
        "FedShield capture session ended."
    )