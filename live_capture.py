import torch
import torch.nn as nn
import numpy as np
import joblib
import subprocess
import sqlite3
import shap
import os
import threading
from collections import defaultdict, deque
from scapy.all import sniff, IP, TCP, UDP
import time
import json
import uuid

BASE = os.path.dirname(os.path.abspath(__file__))

from llm_incident_report import generate_incident_report, init_incident_reports_table
from online_retrain import (
    init_retrain_buffer, log_retrain_sample, should_retrain,
    run_incremental_retrain, VERSION_PATH
)

CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

# Standard NSL-KDD 41-feature ordering (used for SHAP feature name labeling)
FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

class MultiClassIDS(nn.Module):
    def __init__(self, input_dim=41, num_classes=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.network(x)

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(BASE, "models", "fedshield_logs.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            src       TEXT,
            dst       TEXT,
            proto     TEXT,
            prediction TEXT,
            confidence REAL,
            tag       TEXT,
            blocked   INTEGER DEFAULT 0,
            incident_id TEXT
        )
    """)
    # Migration: if this DB file already existed from before the incident_id
    # column was added, CREATE TABLE IF NOT EXISTS above is a no-op — add the
    # column manually in that case.
    existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(detections)")]
    if "incident_id" not in existing_cols:
        conn.execute("ALTER TABLE detections ADD COLUMN incident_id TEXT")
        print("Migrated existing detections table: added incident_id column")
    conn.commit()
    return conn

db_conn = init_db()
print("SQLite database initialised: models/fedshield_logs.db")

# Incident reports table lives in the same DB file now
init_incident_reports_table()
print("Incident reports table ready (same DB)")

init_retrain_buffer()

# Track the model version we've loaded, so we can detect when online_retrain.py
# has committed a new one and hot-reload without restarting the capture process.
current_model_version = None
if os.path.exists(VERSION_PATH):
    try:
        with open(VERSION_PATH) as f:
            current_model_version = json.load(f).get("version")
    except Exception:
        pass

retrain_lock = threading.Lock()
retrain_in_progress = False

def maybe_trigger_retrain():
    """Runs the retrain check off the packet-capture thread so a fine-tune (which
    can take a few seconds) never stalls live sniffing."""
    global retrain_in_progress
    with retrain_lock:
        if retrain_in_progress:
            return
        retrain_in_progress = True

    def _worker():
        global retrain_in_progress
        try:
            if should_retrain():
                print("\n🔄 [online_retrain] Trigger condition met — starting incremental fine-tune...")
                result = run_incremental_retrain()
                if result.get("accepted"):
                    print(f"🔄 [online_retrain] Update accepted and saved. "
                          f"F1 {result['f1_before']} -> {result['f1_after']}\n")
                else:
                    print(f"🔄 [online_retrain] Update not applied: {result.get('reason', result.get('error'))}\n")
        finally:
            with retrain_lock:
                retrain_in_progress = False

    threading.Thread(target=_worker, daemon=True).start()

def maybe_hot_reload_model():
    """Checks whether online_retrain.py has committed a newer model version and,
    if so, reloads weights into the already-running model object in place —
    no process restart required."""
    global current_model_version
    if not os.path.exists(VERSION_PATH):
        return
    try:
        with open(VERSION_PATH) as f:
            version_info = json.load(f)
        new_version = version_info.get("version")
        if new_version and new_version != current_model_version:
            model.load_state_dict(torch.load(os.path.join(BASE, "models", "federated_noniid_model.pth"), map_location="cpu"))
            model.eval()
            current_model_version = new_version
            print(f"\n♻️  [online_retrain] Hot-reloaded model to version {new_version} "
                  f"(round {version_info.get('round')}) — no restart needed.\n")
    except Exception as e:
        print(f"   [WARN] Hot reload check failed: {e}")

# ── Model + scaler ────────────────────────────────────────────────────────────
model = MultiClassIDS()
model.load_state_dict(torch.load(os.path.join(BASE, "models", "federated_noniid_model.pth"), map_location="cpu"))
model.eval()

scaler = joblib.load(os.path.join(BASE, "models", "scaler_multiclass.pkl"))
print("Model + scaler loaded (federated_noniid_model.pth + scaler_multiclass.pkl)")

# ── SHAP explainer (initialized once, reused for every detection) ────────────
# Background sample drawn from training data so SHAP has a baseline to compare against.
try:
    _background_raw = np.load(os.path.join(BASE, "data", "X_train_mc.npy"))[:100]
    _background_t = torch.FloatTensor(_background_raw)
    shap_explainer = shap.DeepExplainer(model, _background_t)
    print("SHAP DeepExplainer initialised (background: 100 samples from X_train_mc.npy)")
except Exception as e:
    shap_explainer = None
    print(f"[WARN] Could not initialise SHAP explainer, incident reports will skip feature attribution: {e}")

def get_top_shap_features(x_tensor, pred_class, top_n=3):
    """Returns [(feature_name, shap_value), ...] for the predicted class, or [] on failure."""
    if shap_explainer is None:
        return []
    try:
        shap_values = shap_explainer.shap_values(x_tensor)
        # shap_values shape depends on SHAP version: list-per-class or (1, features, classes)
        if isinstance(shap_values, list):
            class_values = shap_values[pred_class][0]
        else:
            class_values = shap_values[0, :, pred_class]
        top_idx = np.argsort(np.abs(class_values))[::-1][:top_n]
        return [(FEATURE_NAMES[i], float(class_values[i])) for i in top_idx]
    except Exception as e:
        print(f"   [WARN] SHAP computation failed: {e}")
        return []

# ── Rate-limiting for LLM incident reports ────────────────────────────────────
# Groq's free tier is 30 requests/min — never call it per-packet. Only generate
# one report per distinct (src_ip, predicted_class) pair per session, plus always
# on a fresh port-scan alert (which is already deduped via alerted_pairs below).
reported_incidents = set()

# ── Sliding-window state ──────────────────────────────────────────────────────
WINDOW_SIZE = 100
recent_connections = deque(maxlen=WINDOW_SIZE)
host_stats = defaultdict(lambda: {"count": 0, "serror": 0, "same_srv": 0, "ports": set()})

PROTO_MAP = {"tcp": 1, "udp": 2, "icmp": 0}
log = []

# ── Burst port-scan detector ──────────────────────────────────────────────────
port_scan_tracker = defaultdict(lambda: {"ports": set(), "first_seen": None})
SCAN_PORT_THRESHOLD = 8
SCAN_WINDOW_SECONDS = 3
alerted_pairs = set()

# ── Auto-block ────────────────────────────────────────────────────────────────
blocked_ips = set()

def block_ip(ip):
    """Add a Windows Firewall rule to block all inbound traffic from this IP."""
    if ip in blocked_ips:
        return False
    try:
        rule_name = f"FedShield_Block_{ip.replace('.', '_')}"
        cmd = [
            "netsh", "advfirewall", "firewall", "add", "rule",
            f"name={rule_name}",
            "dir=in", "action=block",
            f"remoteip={ip}"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            blocked_ips.add(ip)
            return True
        else:
            print(f"   [WARN] Could not block {ip}: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"   [WARN] Block failed for {ip}: {e}")
        return False

def check_port_scan(src, dst, dport, proto):
    if proto != "tcp" or dport is None:
        return False
    key = (src, dst)
    now = time.time()
    entry = port_scan_tracker[key]
    if entry["first_seen"] is None or (now - entry["first_seen"]) > SCAN_WINDOW_SECONDS:
        entry["first_seen"] = now
        entry["ports"] = set()
    entry["ports"].add(dport)
    if len(entry["ports"]) >= SCAN_PORT_THRESHOLD and key not in alerted_pairs:
        alerted_pairs.add(key)
        return True
    return False

def extract_features(pkt):
    if IP not in pkt:
        return None, None

    features = np.zeros(41, dtype=np.float32)
    proto = "tcp" if TCP in pkt else ("udp" if UDP in pkt else "icmp")
    dst   = pkt[IP].dst
    src   = pkt[IP].src
    dport = pkt[TCP].dport if TCP in pkt else None

    features[1]  = PROTO_MAP.get(proto, 0)
    features[4]  = len(pkt)
    features[5]  = 0
    features[11] = 0

    recent_connections.append(dst)
    stats = host_stats[dst]
    stats["count"] += 1
    if dport:
        stats["ports"].add(dport)

    is_syn_error = False
    if TCP in pkt and pkt[TCP].flags == "S":
        is_syn_error = True
        stats["serror"] += 1

    same_host_count = sum(1 for d in recent_connections if d == dst)
    serror_rate     = stats["serror"] / max(stats["count"], 1)

    features[22] = same_host_count
    features[24] = serror_rate
    features[37] = serror_rate
    features[28] = same_host_count / max(len(recent_connections), 1)
    features[31] = min(stats["count"], 255)

    return features, {
        "src": src, "dst": dst, "proto": proto,
        "is_syn": is_syn_error, "count": stats["count"], "dport": dport
    }

def maybe_generate_incident_report(meta, label, confidence, x_tensor, pred_class, force=False):
    """Generates an LLM incident report, but only once per (src, label) pair per session
    unless force=True (used for port-scan alerts, which are already deduped upstream)."""
    key = (meta["src"], label)
    if not force and key in reported_incidents:
        return None
    reported_incidents.add(key)

    shap_features = get_top_shap_features(x_tensor, pred_class)
    incident_id = str(uuid.uuid4())

    flow_stats = {
        "src_ip": meta["src"],
        "dst_ip": meta["dst"],
        "dst_port": meta["dport"] if meta["dport"] else "N/A",
        "protocol": meta["proto"],
        "packet_count_this_host": meta["count"],
    }

    report_text = generate_incident_report(
        incident_id=incident_id,
        flow_stats=flow_stats,
        prediction=label,
        confidence=confidence,
        shap_features=shap_features
    )
    print(f"\n📋 INCIDENT REPORT [{incident_id[:8]}]:\n{report_text}\n")
    return incident_id

def classify_packet(pkt):
    raw_features, meta = extract_features(pkt)
    if raw_features is None:
        return

    scaled = scaler.transform(raw_features.reshape(1, -1))
    x = torch.FloatTensor(scaled)

    with torch.no_grad():
        logits     = model(x)
        probs      = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    label = CLASS_NAMES[pred_class]
    scan_detected = check_port_scan(meta["src"], meta["dst"], meta["dport"], meta["proto"])

    entry = {
        "time": time.strftime("%H:%M:%S"),
        "src": meta["src"], "dst": meta["dst"], "proto": meta["proto"],
        "prediction": label, "confidence": round(confidence, 3),
        "tag": "ATTACK" if pred_class != 0 else "normal"
    }
    log.append(entry)

    incident_id = None
    if scan_detected:
        incident_id = maybe_generate_incident_report(meta, label, confidence, x, pred_class, force=True)
    elif pred_class != 0:
        incident_id = maybe_generate_incident_report(meta, label, confidence, x, pred_class)

    # ── Write to SQLite ───────────────────────────────────────────────────────
    db_conn.execute("""
        INSERT INTO detections (timestamp, src, dst, proto, prediction, confidence, tag, incident_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (entry['time'], entry['src'], entry['dst'], entry['proto'],
          entry['prediction'], entry['confidence'], entry['tag'], incident_id))
    db_conn.commit()

    # ── Port-scan alert + auto-block ──────────────────────────────────────────
    if scan_detected:
        print(f"\n🚨🚨🚨 [{entry['time']}] PORT SCAN DETECTED: {meta['src']} -> {meta['dst']} "
              f"({len(port_scan_tracker[(meta['src'], meta['dst'])]['ports'])} ports in "
              f"{SCAN_WINDOW_SECONDS}s) => Probe/Reconnaissance Attack 🚨🚨🚨")

        blocked = block_ip(meta['src'])
        if blocked:
            db_conn.execute(
                "UPDATE detections SET blocked=1 WHERE src=? ORDER BY id DESC LIMIT 1",
                (meta['src'],)
            )
            db_conn.commit()
            print(f"🛡️  AUTO-BLOCKED: {meta['src']} via Windows Firewall — all inbound traffic now denied\n")
        else:
            print(f"   [{meta['src']} already blocked or block failed]\n")

        # ── Online retraining: log a rule-confirmed sample ─────────────────────
        # Label comes from check_port_scan() — a fixed heuristic independent of the
        # model's own prediction — NOT from pred_class. Using the model's own output
        # as its own training label would just reinforce whatever it already believes.
        PROBE_CLASS_INDEX = CLASS_NAMES.index("Probe")
        log_retrain_sample(raw_features, PROBE_CLASS_INDEX, source="rule_confirmed_probe")

    elif pred_class != 0:
        print(f"🚨 [{entry['time']}] {meta['src']} -> {meta['dst']} ({meta['proto']}) "
              f"=> {label} (confidence: {confidence:.2%})")
    else:
        print(f"   [{entry['time']}] {meta['src']} -> {meta['dst']} ({meta['proto']}) => normal "
              f"(confidence: {confidence:.2%})")

    # ── JSON snapshot every 5 packets (kept for dashboard compatibility) ───────
    if len(log) % 5 == 0:
        with open(os.path.join(BASE, "models", "live_log.json"), "w") as f:
            json.dump(log[-200:], f)

    # ── Online retraining: periodic trigger + hot-reload checks ────────────────
    if len(log) % 25 == 0:
        maybe_hot_reload_model()
        maybe_trigger_retrain()

# ── Entry point ───────────────────────────────────────────────────────────────
print("\n===== FedShield Live Capture Started =====")
print(f"Port scan detection : {SCAN_PORT_THRESHOLD}+ distinct ports in {SCAN_WINDOW_SECONDS}s")
print("Auto-block          : ENABLED (Windows Firewall)")
print("Audit log           : models/fedshield_logs.db  (SQLite)")
print("Incident reports    : ENABLED (Groq LLM, rate-limited per src+attack-type)")
print("Sniffing real network traffic. Press Ctrl+C to stop.\n")

try:
    sniff(prn=classify_packet, store=False, count=0)
except KeyboardInterrupt:
    print("\nCapture stopped.")
    with open(os.path.join(BASE, "models", "live_log.json"), "w") as f:
        json.dump(log[-200:], f)
    total = db_conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    attacks = db_conn.execute("SELECT COUNT(*) FROM detections WHERE tag='ATTACK'").fetchone()[0]
    blocked_count = db_conn.execute("SELECT COUNT(*) FROM detections WHERE blocked=1").fetchone()[0]
    reports_count = db_conn.execute("SELECT COUNT(*) FROM incident_reports").fetchone()[0]
    print(f"\nSession summary:")
    print(f"  Total packets logged : {total}")
    print(f"  Attacks detected     : {attacks}")
    print(f"  IPs auto-blocked     : {blocked_count}")
    print(f"  Incident reports     : {reports_count}")
    print(f"  DB                   : models/fedshield_logs.db")
    if blocked_ips:
        print(f"  Blocked IPs          : {blocked_ips}")