import torch
import torch.nn as nn
import numpy as np
import joblib
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

# Load trained model + the SAME scaler used during training
model = MultiClassIDS()
model.load_state_dict(torch.load("models/federated_noniid_model.pth"))
model.eval()

scaler = joblib.load("models/scaler_multiclass.pkl")
print("Model + scaler loaded (federated_noniid_model.pth + scaler_multiclass.pkl)")

# Sliding window state — tracks recent connections per destination host
WINDOW_SIZE = 100
recent_connections = deque(maxlen=WINDOW_SIZE)
host_stats = defaultdict(lambda: {"count": 0, "serror": 0, "same_srv": 0})

PROTO_MAP = {"tcp": 1, "udp": 2, "icmp": 0}

log = []

def extract_features(pkt):
    """Build a raw (unscaled) 41-dim feature vector approximating NSL-KDD fields."""
    if IP not in pkt:
        return None, None

    features = np.zeros(41, dtype=np.float32)

    proto = "tcp" if TCP in pkt else ("udp" if UDP in pkt else "icmp")
    dst = pkt[IP].dst
    src = pkt[IP].src

    features[1] = PROTO_MAP.get(proto, 0)       # protocol_type
    features[4] = len(pkt)                       # src_bytes (approx via pkt length)
    features[5] = 0                               # dst_bytes (unknown, no flow tracking)
    features[11] = 0                               # logged_in (unknown from raw packet)

    recent_connections.append(dst)
    stats = host_stats[dst]
    stats["count"] += 1

    is_syn_error = False
    if TCP in pkt:
        flags = pkt[TCP].flags
        if flags == "S":
            is_syn_error = True
            stats["serror"] += 1

    same_host_count = sum(1 for d in recent_connections if d == dst)
    features[22] = same_host_count                                   # count
    serror_rate = stats["serror"] / max(stats["count"], 1)
    features[24] = serror_rate                                       # serror_rate
    features[37] = serror_rate                                       # dst_host_serror_rate
    features[28] = same_host_count / max(len(recent_connections), 1) # same_srv_rate
    features[31] = min(stats["count"], 255)                          # dst_host_count

    return features, {
        "src": src, "dst": dst, "proto": proto,
        "is_syn": is_syn_error, "count": stats["count"]
    }

def classify_packet(pkt):
    raw_features, meta = extract_features(pkt)
    if raw_features is None:
        return

    # CRITICAL FIX: scale using the SAME scaler fit during training
    scaled = scaler.transform(raw_features.reshape(1, -1))
    x = torch.FloatTensor(scaled)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    label = CLASS_NAMES[pred_class]
    tag = "ATTACK" if pred_class != 0 else "normal"

    entry = {
        "time": time.strftime("%H:%M:%S"),
        "src": meta["src"], "dst": meta["dst"], "proto": meta["proto"],
        "prediction": label, "confidence": round(confidence, 3), "tag": tag
    }
    log.append(entry)

    if pred_class != 0:
        print(f"🚨 [{entry['time']}] {meta['src']} -> {meta['dst']} ({meta['proto']}) "
              f"=> {label} (confidence: {confidence:.2%})")
    else:
        print(f"   [{entry['time']}] {meta['src']} -> {meta['dst']} ({meta['proto']}) => normal "
              f"(confidence: {confidence:.2%})")

    if len(log) % 5 == 0:
        with open("models/live_log.json", "w") as f:
            json.dump(log[-200:], f)

print("\n===== FedShield Live Capture Started =====")
print("Sniffing real network traffic on this machine. Press Ctrl+C to stop.\n")

try:
    sniff(prn=classify_packet, store=False, count=0)
except KeyboardInterrupt:
    print("\nCapture stopped.")
    with open("models/live_log.json", "w") as f:
        json.dump(log[-200:], f)
    print(f"Saved {len(log)} classified packets to models/live_log.json")