"""
FedShield — Live Packet Capture

Pipeline:

Packet
  ↓
41 NSL-KDD-compatible features
  ↓
Saved categorical encoders
  ↓
Saved StandardScaler
  ↓
MultiClassIDS
  ↓
Normal / DoS / Probe / R2L / U2R
  ↓
Rule confirmation
  ↓
SHAP explanation
  ↓
Incident report
  ↓
Optional firewall response
  ↓
Online retraining buffer
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

import joblib
import numpy as np
import shap
import torch

from scapy.all import (
    sniff,
    IP,
    TCP,
    UDP
)

from model import MultiClassIDS

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

BASE = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE,
    "models",
    "federated_noniid_model.pth"
)

SCALER_PATH = os.path.join(
    BASE,
    "models",
    "scaler_multiclass.pkl"
)

ENCODERS_PATH = os.path.join(
    BASE,
    "models",
    "encoders_multiclass.pkl"
)

DB_PATH = os.path.join(
    BASE,
    "models",
    "fedshield_logs.db"
)

LIVE_LOG_PATH = os.path.join(
    BASE,
    "models",
    "live_log.json"
)

TRAIN_BG_PATH = os.path.join(
    BASE,
    "data",
    "X_train_mc.npy"
)


# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = [
    "Normal",
    "DoS",
    "Probe",
    "R2L",
    "U2R"
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


# ============================================================
# THRESHOLDS
# ============================================================

ML_ATTACK_THRESHOLD = 0.92

ML_SUSPICIOUS_THRESHOLD = 0.70

ML_CONFIRMATIONS_REQUIRED = 3

ML_CONFIRMATION_WINDOW = 10

SCAN_PORT_THRESHOLD = 8

SCAN_WINDOW_SECONDS = 3

PROBE_CONFIDENCE_THRESHOLD = 0.85

WINDOW_SECONDS = 10


# ============================================================
# DIRECTORY
# ============================================================

os.makedirs(
    os.path.join(
        BASE,
        "models"
    ),
    exist_ok=True
)


# ============================================================
# DATABASE
# ============================================================

def init_db():

    with sqlite3.connect(
        DB_PATH,
        timeout=10
    ) as conn:

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


def db_insert_detection(
    entry
):

    with sqlite3.connect(
        DB_PATH,
        timeout=10
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
            )
        )

        return cursor.lastrowid


def mark_blocked(
    row_id
):

    if row_id is None:
        return

    with sqlite3.connect(
        DB_PATH,
        timeout=10
    ) as conn:

        conn.execute(
            """
            UPDATE detections
            SET blocked=1
            WHERE id=?
            """,
            (row_id,)
        )


init_db()
init_incident_reports_table()
init_retrain_buffer()


# ============================================================
# MODEL
# ============================================================

if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(
        f"Missing model:\n"
        f"{MODEL_PATH}\n\n"
        "Run federated_noniid.py first."
    )


model = MultiClassIDS(
    input_dim=41,
    num_classes=5
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

print(
    "Multi-class federated model loaded."
)


# ============================================================
# SCALER
# ============================================================

if not os.path.exists(
    SCALER_PATH
):

    raise FileNotFoundError(
        f"Missing scaler:\n"
        f"{SCALER_PATH}\n\n"
        "Run preprocess_multiclass.py first."
    )


scaler = joblib.load(
    SCALER_PATH
)


# ============================================================
# ENCODERS
# ============================================================

if not os.path.exists(
    ENCODERS_PATH
):

    raise FileNotFoundError(
        f"Missing encoders:\n"
        f"{ENCODERS_PATH}\n\n"
        "Run preprocess_multiclass.py first."
    )


encoders = joblib.load(
    ENCODERS_PATH
)


required_encoders = {
    "protocol_type",
    "service",
    "flag"
}

missing_encoders = (
    required_encoders
    -
    set(encoders.keys())
)

if missing_encoders:

    raise ValueError(
        "Missing categorical encoders: "
        +
        ", ".join(
            sorted(
                missing_encoders
            )
        )
    )


protocol_encoder = encoders[
    "protocol_type"
]

service_encoder = encoders[
    "service"
]

flag_encoder = encoders[
    "flag"
]


print(
    "Training categorical encoders loaded."
)


# ============================================================
# MODEL VERSION
# ============================================================

current_model_version = None


def read_model_version():

    if not os.path.exists(
        VERSION_PATH
    ):

        return None

    try:

        with open(
            VERSION_PATH,
            encoding="utf-8"
        ) as f:

            return json.load(
                f
            ).get(
                "version"
            )

    except Exception:

        return None


current_model_version = (
    read_model_version()
)


# ============================================================
# SHAP
# ============================================================

shap_explainer = None

try:

    if os.path.exists(
        TRAIN_BG_PATH
    ):

        background = np.load(
            TRAIN_BG_PATH
        )[:100]

        background_tensor = torch.as_tensor(
            background,
            dtype=torch.float32
        )

        shap_explainer = shap.DeepExplainer(
            model,
            background_tensor
        )

        print(
            "SHAP DeepExplainer initialised."
        )

except Exception as e:

    print(
        f"[WARN] SHAP disabled: {e}"
    )


def get_top_shap_features(
    x_tensor,
    pred_class,
    top_n=3
):

    if shap_explainer is None:

        return []

    try:

        values = (
            shap_explainer.shap_values(
                x_tensor
            )
        )

        if isinstance(
            values,
            list
        ):

            if pred_class >= len(
                values
            ):

                return []

            class_values = np.asarray(
                values[pred_class]
            )[0]

        else:

            arr = np.asarray(
                values
            )

            if arr.ndim == 3:

                # (samples, features, classes)
                if (
                    arr.shape[0] == 1
                    and
                    arr.shape[2] == 5
                ):

                    class_values = arr[
                        0,
                        :,
                        pred_class
                    ]

                # (classes, samples, features)
                elif (
                    arr.shape[0] == 5
                    and
                    arr.shape[1] == 1
                ):

                    class_values = arr[
                        pred_class,
                        0,
                        :
                    ]

                else:

                    return []

            elif arr.ndim == 2:

                class_values = arr[0]

            else:

                return []

        class_values = np.asarray(
            class_values
        ).reshape(-1)

        if len(
            class_values
        ) != 41:

            return []

        indices = np.argsort(
            np.abs(
                class_values
            )
        )[::-1][:top_n]

        return [
            (
                FEATURE_NAMES[i],
                float(
                    class_values[i]
                )
            )
            for i in indices
        ]

    except Exception as e:

        print(
            f"[WARN] SHAP computation failed: "
            f"{e}"
        )

        return []


# ============================================================
# STATE
# ============================================================

reported_incidents = set()

blocked_ips = set()

alerted_pairs = set()

attack_evidence = defaultdict(
    deque
)

recent_packets = deque(
    maxlen=5000
)

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
    }
)

host_stats = defaultdict(
    lambda: {
        "timestamps": deque(
            maxlen=500
        ),
        "dsts": deque(
            maxlen=500
        ),
        "ports": deque(
            maxlen=500
        ),
        "services": deque(
            maxlen=500
        ),
        "errors": 0,
        "syn": 0,
        "packets": 0,
    }
)

log = []

retrain_lock = threading.Lock()

retrain_in_progress = False


# ============================================================
# ENCODING
# ============================================================

def encode_category(
    encoder,
    value,
    fallback=None
):

    value = str(
        value
    ).strip()

    classes = list(
        encoder.classes_
    )

    if value in classes:

        return int(
            encoder.transform(
                [value]
            )[0]
        )

    if fallback is not None:

        fallback = str(
            fallback
        ).strip()

        if fallback in classes:

            return int(
                encoder.transform(
                    [fallback]
                )[0]
            )

    # Deterministic unknown-category fallback.
    return 0


# ============================================================
# SERVICE MAPPING
# ============================================================

PORT_SERVICE_MAP = {

    20: "ftp_data",
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    53: "domain",
    80: "http",
    110: "pop_3",
    111: "sunrpc",
    119: "nntp",
    135: "msrpc",
    139: "netbios_ssn",
    143: "imap4",
    443: "http",
    993: "imap4",
    995: "pop_3",
    1433: "sql_net",
    3306: "mysql",
    3389: "remote_job",
    8080: "http",
}


def infer_service(
    dport
):

    if dport is None:

        return encode_category(
            service_encoder,
            "other"
        )

    service_name = PORT_SERVICE_MAP.get(
        int(dport),
        "other"
    )

    return encode_category(
        service_encoder,
        service_name,
        fallback="other"
    )


# ============================================================
# PACKET HELPERS
# ============================================================

def get_protocol(
    pkt
):

    if TCP in pkt:
        return "tcp"

    if UDP in pkt:
        return "udp"

    return "icmp"


def get_destination_port(
    pkt
):

    if TCP in pkt:

        return int(
            pkt[TCP].dport
        )

    if UDP in pkt:

        return int(
            pkt[UDP].dport
        )

    return None


def get_source_port(
    pkt
):

    if TCP in pkt:

        return int(
            pkt[TCP].sport
        )

    if UDP in pkt:

        return int(
            pkt[UDP].sport
        )

    return None


def get_tcp_flags(
    pkt
):

    if TCP not in pkt:

        return ""

    try:

        return str(
            pkt[TCP].flags
        )

    except Exception:

        return ""


# ============================================================
# STATE CLEANUP
# ============================================================

def cleanup_state():

    now = time.monotonic()

    while recent_packets:

        if (
            now -
            recent_packets[0]["time"]
            >
            WINDOW_SECONDS
        ):

            recent_packets.popleft()

        else:

            break

    for host, stats in list(
        host_stats.items()
    ):

        timestamps = stats[
            "timestamps"
        ]

        while timestamps:

            if (
                now -
                timestamps[0]
                >
                WINDOW_SECONDS
            ):

                timestamps.popleft()

            else:

                break

        if not timestamps:

            host_stats.pop(
                host,
                None
            )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    pkt
):

    if IP not in pkt:

        return None, None

    cleanup_state()

    now = time.monotonic()

    proto = get_protocol(
        pkt
    )

    src = pkt[IP].src
    dst = pkt[IP].dst

    dport = get_destination_port(
        pkt
    )

    sport = get_source_port(
        pkt
    )

    packet_length = int(
        len(pkt)
    )

    flags = get_tcp_flags(
        pkt
    )

    protocol_code = encode_category(
        protocol_encoder,
        proto
    )

    service_code = infer_service(
        dport
    )

    flag_code = encode_category(
        flag_encoder,
        flags,
        fallback="OTH"
    )

    flow_key = (
        src,
        dst,
        sport,
        dport,
        proto
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

    host = host_stats[
        src
    ]

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

    if (
        TCP in pkt
        and
        "S" in flags
        and
        "A" not in flags
    ):

        host["syn"] += 1

    if (
        TCP in pkt
        and
        "R" in flags
    ):

        host["errors"] += 1

    count = len(
        host["timestamps"]
    )

    unique_dsts = len(
        set(
            host["dsts"]
        )
    )

    unique_ports = len(
        set(
            host["ports"]
        )
    )

    unique_services = len(
        set(
            host["services"]
        )
    )

    syn_count = host[
        "syn"
    ]

    error_count = host[
        "errors"
    ]

    serror_rate = (
        syn_count /
        max(
            count,
            1
        )
    )

    rerror_rate = (
        error_count /
        max(
            count,
            1
        )
    )

    same_dst_count = sum(
        1
        for item in host["dsts"]
        if item == dst
    )

    same_service_count = sum(
        1
        for item in host["services"]
        if item == service_code
    )

    same_dst_rate = (
        same_dst_count /
        max(
            count,
            1
        )
    )

    same_service_rate = (
        same_service_count /
        max(
            len(
                host["services"]
            ),
            1
        )
    )

    diff_service_rate = (
        1.0 -
        same_service_rate
    )

    destination_packets = 0
    destination_services = 0
    destination_same_source = 0
    destination_errors = 0

    for item in recent_packets:

        if item["dst"] != dst:
            continue

        destination_packets += 1

        if (
            item["service"]
            ==
            service_code
        ):

            destination_services += 1

        if item["src"] == src:

            destination_same_source += 1

        if item["error"]:

            destination_errors += 1

    dst_host_count = (
        destination_packets
    )

    dst_host_srv_count = (
        destination_services
    )

    dst_host_same_srv_rate = (
        destination_services /
        max(
            destination_packets,
            1
        )
    )

    dst_host_diff_srv_rate = (
        1.0 -
        dst_host_same_srv_rate
    )

    dst_host_same_src_port_rate = (
        destination_same_source /
        max(
            destination_packets,
            1
        )
    )

    dst_host_srv_diff_host_rate = (
        1.0 -
        dst_host_same_src_port_rate
    )

    dst_host_serror_rate = (
        destination_errors /
        max(
            destination_packets,
            1
        )
    )

    dst_host_srv_serror_rate = (
        dst_host_serror_rate
    )

    dst_host_rerror_rate = (
        destination_errors /
        max(
            destination_packets,
            1
        )
    )

    dst_host_srv_rerror_rate = (
        dst_host_rerror_rate
    )

    # --------------------------------------------------------
    # 41 FEATURES
    # --------------------------------------------------------

    features = np.zeros(
        41,
        dtype=np.float32
    )

    features[0] = min(
        now -
        flow["first_seen"],
        3600.0
    )

    features[1] = protocol_code

    features[2] = service_code

    features[3] = flag_code

    features[4] = min(
        flow["src_bytes"],
        10_000_000
    )

    features[5] = min(
        packet_length,
        10_000_000
    )

    features[6] = float(
        src == dst
    )

    features[7] = float(
        getattr(
            pkt[IP],
            "frag",
            0
        ) > 0
    )

    features[8] = float(
        TCP in pkt
        and
        getattr(
            pkt[TCP],
            "urgptr",
            0
        ) > 0
    )

    features[9] = min(
        unique_services,
        255
    )

    features[10] = 0.0

    features[11] = float(
        TCP in pkt
        and
        "A" in flags
    )

    features[12:22] = 0.0

    features[22] = min(
        count,
        255
    )

    features[23] = min(
        same_service_count,
        255
    )

    features[24] = serror_rate
    features[25] = serror_rate

    features[26] = rerror_rate
    features[27] = rerror_rate

    features[28] = same_service_rate

    features[29] = diff_service_rate

    features[30] = (
        1.0 -
        same_dst_rate
    )

    features[31] = min(
        dst_host_count,
        255
    )

    features[32] = min(
        dst_host_srv_count,
        255
    )

    features[33] = (
        dst_host_same_srv_rate
    )

    features[34] = (
        dst_host_diff_srv_rate
    )

    features[35] = (
        dst_host_same_src_port_rate
    )

    features[36] = (
        dst_host_srv_diff_host_rate
    )

    features[37] = (
        dst_host_serror_rate
    )

    features[38] = (
        dst_host_srv_serror_rate
    )

    features[39] = (
        dst_host_rerror_rate
    )

    features[40] = (
        dst_host_srv_rerror_rate
    )

    recent_packets.append(
        {
            "time": now,
            "src": src,
            "dst": dst,
            "proto": proto,
            "sport": sport,
            "dport": dport,
            "service": service_code,
            "error": (
                TCP in pkt
                and
                "R" in flags
            )
        }
    )

    meta = {

        "src": src,
        "dst": dst,
        "proto": proto,
        "sport": sport,
        "dport": dport,

        "service":
            service_code,

        "flag":
            flags,

        "count":
            count,

        "unique_ports":
            unique_ports,

        "unique_dsts":
            unique_dsts,

        "unique_services":
            unique_services,

        "syn_count":
            syn_count,

        "error_count":
            error_count,

        "flow_packets":
            flow["packets"],

        "flow_duration":
            now -
            flow["first_seen"]
    }

    return features, meta


# ============================================================
# ATTACK EVIDENCE
# ============================================================

def register_attack_evidence(
    src,
    dst,
    label,
    confidence
):

    now = time.monotonic()

    key = (
        src,
        dst,
        label
    )

    evidence = attack_evidence[
        key
    ]

    while evidence:

        if (
            now -
            evidence[0]
            >
            ML_CONFIRMATION_WINDOW
        ):

            evidence.popleft()

        else:

            break

    if (
        confidence
        >=
        ML_ATTACK_THRESHOLD
    ):

        evidence.append(
            now
        )

    return len(
        evidence
    )


# ============================================================
# PORT SCAN DETECTION
# ============================================================

port_scan_tracker = defaultdict(
    lambda: {
        "ports": set(),
        "first_seen": None
    }
)


def check_port_scan(
    src,
    dst,
    dport,
    proto
):

    if (
        proto != "tcp"
        or
        dport is None
    ):

        return False

    key = (
        src,
        dst
    )

    now = time.monotonic()

    entry = port_scan_tracker[
        key
    ]

    if (
        entry["first_seen"] is None
        or
        now -
        entry["first_seen"]
        >
        SCAN_WINDOW_SECONDS
    ):

        entry["first_seen"] = now
        entry["ports"] = set()

    entry["ports"].add(
        int(dport)
    )

    if (
        len(
            entry["ports"]
        )
        >=
        SCAN_PORT_THRESHOLD
        and
        key not in alerted_pairs
    ):

        alerted_pairs.add(
            key
        )

        return True

    return False


# ============================================================
# FIREWALL
# ============================================================

def block_ip(
    ip
):

    import ipaddress

    try:

        ipaddress.ip_address(
            ip
        )

    except ValueError:

        return False

    if ip in blocked_ips:

        return False

    rule_name = (
        "FedShield_Block_"
        +
        ip.replace(
            ".",
            "_"
        )
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
        f"remoteip={ip}"
    ]

    try:

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=False,
            timeout=10
        )

        if result.returncode == 0:

            blocked_ips.add(
                ip
            )

            return True

        print(
            "[WARN] Firewall block failed: "
            +
            result.stderr.strip()
        )

    except Exception as e:

        print(
            f"[WARN] Firewall error: {e}"
        )

    return False


# ============================================================
# INCIDENT REPORT
# ============================================================

def maybe_generate_incident_report(
    meta,
    label,
    confidence,
    x_tensor,
    pred_class,
    force=False
):

    key = (
        meta["src"],
        meta["dst"],
        label
    )

    if (
        not force
        and
        key in reported_incidents
    ):

        return None

    reported_incidents.add(
        key
    )

    incident_id = str(
        uuid.uuid4()
    )

    flow_stats = {

        "src_ip":
            meta["src"],

        "dst_ip":
            meta["dst"],

        "src_port":
            meta["sport"],

        "dst_port":
            meta["dport"],

        "protocol":
            meta["proto"],

        "packet_count_this_host":
            meta["count"],

        "flow_packets":
            meta["flow_packets"],

        "flow_duration":
            round(
                meta["flow_duration"],
                3
            ),

        "unique_ports":
            meta["unique_ports"],

        "unique_destinations":
            meta["unique_dsts"],

        "syn_count":
            meta["syn_count"],

        "error_count":
            meta["error_count"]
    }

    shap_features = (
        get_top_shap_features(
            x_tensor,
            pred_class
        )
    )

    report = generate_incident_report(
        incident_id,
        flow_stats,
        label,
        confidence,
        shap_features
    )

    print(
        f"\nINCIDENT REPORT "
        f"[{incident_id[:8]}]:\n"
        f"{report}\n"
    )

    return incident_id


# ============================================================
# ONLINE RETRAIN
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
                    +
                    str(
                        result.get(
                            "reason",
                            result.get(
                                "error",
                                "done"
                            )
                        )
                    )
                )

        except Exception as e:

            print(
                f"[WARN] Online retraining failed: "
                f"{e}"
            )

        finally:

            with retrain_lock:

                retrain_in_progress = False

    threading.Thread(
        target=worker,
        daemon=True
    ).start()


# ============================================================
# HOT RELOAD
# ============================================================

def maybe_hot_reload_model():

    global current_model_version

    version = read_model_version()

    if (
        version is None
        or
        version ==
        current_model_version
    ):

        return

    try:

        state = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(
            state
        )

        model.eval()

        current_model_version = (
            version
        )

        print(
            f"[online_retrain] "
            f"Hot-reloaded model "
            f"version {version}"
        )

    except Exception as e:

        print(
            f"[WARN] Hot reload failed: "
            f"{e}"
        )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_packet(
    pkt
):

    # Check whether online retraining produced
    # a new model before inference.
    maybe_hot_reload_model()

    raw_features, meta = (
        extract_features(
            pkt
        )
    )

    if raw_features is None:

        return

    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    try:

        scaled = scaler.transform(
            raw_features.reshape(
                1,
                -1
            )
        ).astype(
            np.float32
        )

    except Exception as e:

        print(
            f"[WARN] Scaling failed: "
            f"{e}"
        )

        return

    x = torch.as_tensor(
        scaled,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    try:

        with torch.no_grad():

            logits = model(
                x
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            pred_class = int(
                probabilities.argmax(
                    dim=1
                ).item()
            )

            confidence = float(
                probabilities[
                    0,
                    pred_class
                ].item()
            )

    except Exception as e:

        print(
            f"[WARN] Model inference failed: "
            f"{e}"
        )

        return

    label = CLASS_NAMES[
        pred_class
    ]

    # --------------------------------------------------------
    # RULE DETECTION
    # --------------------------------------------------------

    scan_detected = (
        check_port_scan(
            meta["src"],
            meta["dst"],
            meta["dport"],
            meta["proto"]
        )
    )

    incident_id = None

    # --------------------------------------------------------
    # PORT SCAN
    # --------------------------------------------------------

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
                    PROBE_CONFIDENCE_THRESHOLD
                ),
                x,
                probe_class,
                force=True
            )
        )

    # --------------------------------------------------------
    # ML ATTACK
    # --------------------------------------------------------

    elif (
        pred_class != 0
        and
        confidence >=
        ML_ATTACK_THRESHOLD
    ):

        evidence_count = (
            register_attack_evidence(
                meta["src"],
                meta["dst"],
                label,
                confidence
            )
        )

        if (
            evidence_count
            >=
            ML_CONFIRMATIONS_REQUIRED
        ):

            incident_id = (
                maybe_generate_incident_report(
                    meta,
                    label,
                    confidence,
                    x,
                    pred_class
                )
            )

    # --------------------------------------------------------
    # SUSPICIOUS
    # --------------------------------------------------------

    elif (
        pred_class != 0
        and
        confidence >=
        ML_SUSPICIOUS_THRESHOLD
    ):

        print(
            f"[SUSPICIOUS] "
            f"{meta['src']} -> "
            f"{meta['dst']} => "
            f"{label} "
            f"({confidence:.2%})"
        )

    # --------------------------------------------------------
    # TAG
    # --------------------------------------------------------

    confirmed_ml_attack = (
        pred_class != 0
        and
        confidence >=
        ML_ATTACK_THRESHOLD
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
        confidence >=
        ML_SUSPICIOUS_THRESHOLD
    ):

        tag = "SUSPICIOUS"

    else:

        tag = "normal"

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
                4
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
                3
            ),

        "unique_ports":
            meta["unique_ports"]
    }

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    try:

        row_id = db_insert_detection(
            entry
        )

    except Exception as e:

        row_id = None

        print(
            f"[WARN] Database insert failed: "
            f"{e}"
        )

    log.append(
        entry
    )

    # --------------------------------------------------------
    # PORT SCAN RESPONSE
    # --------------------------------------------------------

    if scan_detected:

        print(
            f"\nPORT SCAN DETECTED: "
            f"{meta['src']} -> "
            f"{meta['dst']} "
            f"({SCAN_PORT_THRESHOLD}+ ports/"
            f"{SCAN_WINDOW_SECONDS}s)"
        )

        if block_ip(
            meta["src"]
        ):

            mark_blocked(
                row_id
            )

            print(
                f"AUTO-BLOCKED: "
                f"{meta['src']}"
            )

        # Store the scaled feature vector.
        try:

            log_retrain_sample(
                scaled[0],
                CLASS_NAMES.index(
                    "Probe"
                ),
                "rule_confirmed_probe"
            )

        except Exception as e:

            print(
                f"[WARN] Retraining sample failed: "
                f"{e}"
            )

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
        confidence >=
        ML_SUSPICIOUS_THRESHOLD
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

    # --------------------------------------------------------
    # LOG SNAPSHOT
    # --------------------------------------------------------

    if len(log) % 5 == 0:

        try:

            with open(
                LIVE_LOG_PATH,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    log[-200:],
                    f,
                    indent=2
                )

        except OSError as e:

            print(
                f"[WARN] Live log failed: "
                f"{e}"
            )

    # --------------------------------------------------------
    # RETRAIN CHECK
    # --------------------------------------------------------

    if len(log) % 25 == 0:

        maybe_trigger_retrain()


# ============================================================
# CAPTURE SERVICE
# ============================================================

capture_running = False


def start_capture():
    """Start packet capture when the host permits it.

    Keeping capture behind an explicit function makes this module safe to
    import from FastAPI while preserving the original standalone command.
    """
    global capture_running
    capture_running = True

    print("\n============================================================")
    print("             FedShield Live Capture")
    print("============================================================")
    print("Model: federated_noniid_model.pth")
    print("Model architecture: MultiClassIDS")
    print("Classes: Normal / DoS / Probe / R2L / U2R")
    print("Feature pipeline: 41 NSL-KDD-compatible features")
    print(
        f"Port scan: {SCAN_PORT_THRESHOLD}+ TCP ports in "
        f"{SCAN_WINDOW_SECONDS}s"
    )
    print(f"ML attack threshold: {ML_ATTACK_THRESHOLD:.0%}")
    print(
        f"ML confirmation: {ML_CONFIRMATIONS_REQUIRED} events/"
        f"{ML_CONFIRMATION_WINDOW}s"
    )
    print("Auto-block: confirmed port scans only")
    print("SHAP: enabled when available")
    print("Online retraining: enabled")
    print("Press Ctrl+C to stop.\n")

    try:
        sniff(
            prn=classify_packet,
            store=False,
            count=0
        )
    except KeyboardInterrupt:
        print("\nCapture stopped.")
    except Exception as e:
        print(f"\n[ERROR] Packet capture failed: {e}")
    finally:
        capture_running = False
        try:
            with open(LIVE_LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(log[-200:], f, indent=2)
        except OSError as e:
            print(f"[WARN] Final log write failed: {e}")
        print(f"\nSession packets: {len(log)}")
        print(f"Blocked IPs: {len(blocked_ips)}")
        print("FedShield capture session ended.")


if __name__ == "__main__":
    start_capture()