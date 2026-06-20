import torch
import torch.nn as nn
import numpy as np
import joblib
import subprocess
from collections import defaultdict, deque
from scapy.all import sniff, IP, TCP, UDP
import time
import json

CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

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

model = MultiClassIDS()
model.load_state_dict(torch.load("models/federated_noniid_model.pth"))
model.eval()

scaler = joblib.load("models/scaler_multiclass.pkl")
print("Model + scaler loaded (federated_noniid_model.pth + scaler_multiclass.pkl)")

WINDOW_SIZE = 100
recent_connections = deque(maxlen=WINDOW_SIZE)
host_stats = defaultdict(lambda: {"count": 0, "serror": 0, "same_srv": 0, "ports": set()})

PROTO_MAP = {"tcp": 1, "udp": 2, "icmp": 0}
log = []

# --- Burst-level port scan detector (separate from per-packet model) ---
port_scan_tracker = defaultdict(lambda: {"ports": set(), "first_seen": None})
SCAN_PORT_THRESHOLD = 8       # distinct ports
SCAN_WINDOW_SECONDS = 3       # within this many seconds
alerted_pairs = set()

# --- Auto-block via Windows Firewall ---
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
    dst = pkt[IP].dst
    src = pkt[IP].src
    dport = pkt[TCP].dport if TCP in pkt else None

    features[1] = PROTO_MAP.get(proto, 0)
    features[4] = len(pkt)
    features[5] = 0
    features[11] = 0

    recent_connections.append(dst)
    stats = host_stats[dst]
    stats["count"] += 1
    if dport:
        stats["ports"].add(dport)

    is_syn_error = False
    if TCP in pkt:
        flags = pkt[TCP].flags
        if flags == "S":
            is_syn_error = True
            stats["serror"] += 1

    same_host_count = sum(1 for d in recent_connections if d == dst)
    features[22] = same_host_count
    serror_rate = stats["serror"] / max(stats["count"], 1)
    features[24] = serror_rate
    features[37] = serror_rate
    features[28] = same_host_count / max(len(recent_connections), 1)
    features[31] = min(stats["count"], 255)

    return features, {
        "src": src, "dst": dst, "proto": proto,
        "is_syn": is_syn_error, "count": stats["count"], "dport": dport
    }

def classify_packet(pkt):
    raw_features, meta = extract_features(pkt)
    if raw_features is None:
        return

    scaled = scaler.transform(raw_features.reshape(1, -1))
    x = torch.FloatTensor(scaled)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    label = CLASS_NAMES[pred_class]

    # Burst-level port scan check — independent signal, very reliable
    scan_detected = check_port_scan(meta["src"], meta["dst"], meta["dport"], meta["proto"])

    entry = {
        "time": time.strftime("%H:%M:%S"),
        "src": meta["src"], "dst": meta["dst"], "proto": meta["proto"],
        "prediction": label, "confidence": round(confidence, 3),
        "tag": "ATTACK" if pred_class != 0 else "normal"
    }
    log.append(entry)

    if scan_detected:
        print(f"\n🚨🚨🚨 [{entry['time']}] PORT SCAN DETECTED: {meta['src']} -> {meta['dst']} "
              f"({len(port_scan_tracker[(meta['src'], meta['dst'])]['ports'])} ports in {SCAN_WINDOW_SECONDS}s) "
              f"=> Probe/Reconnaissance Attack 🚨🚨🚨")

        blocked = block_ip(meta['src'])
        if blocked:
            print(f"🛡️  AUTO-BLOCKED: {meta['src']} via Windows Firewall — all inbound traffic now denied\n")
        else:
            print(f"   [{meta['src']} already blocked or block failed]\n")

    elif pred_class != 0:
        print(f"🚨 [{entry['time']}] {meta['src']} -> {meta['dst']} ({meta['proto']}) "
              f"=> {label} (confidence: {confidence:.2%})")
    else:
        print(f"   [{entry['time']}] {meta['src']} -> {meta['dst']} ({meta['proto']}) => normal "
              f"(confidence: {confidence:.2%})")

    if len(log) % 5 == 0:
        with open("models/live_log.json", "w") as f:
            json.dump(log[-200:], f)

print("\n===== FedShield Live Capture Started =====")
print(f"Port scan detection: {SCAN_PORT_THRESHOLD}+ ports in {SCAN_WINDOW_SECONDS}s window")
print("Auto-block: ENABLED (Windows Firewall)")
print("Sniffing real network traffic on this machine. Press Ctrl+C to stop.\n")

try:
    sniff(prn=classify_packet, store=False, count=0)
except KeyboardInterrupt:
    print("\nCapture stopped.")
    with open("models/live_log.json", "w") as f:
        json.dump(log[-200:], f)
    print(f"Saved {len(log)} classified packets to models/live_log.json")
    if blocked_ips:
        print(f"Blocked IPs this session: {blocked_ips}")