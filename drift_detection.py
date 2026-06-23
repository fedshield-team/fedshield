"""
FedShield — Model Drift Detection
Monitors federated model performance over time and alerts when F1 drops
below threshold, indicating concept drift or data distribution shift.

Usage:
    python drift_detection.py          # run analysis
    python drift_detection.py --watch  # continuous monitoring mode

Integrates with:
    - models/fedshield_logs.db  (live detection logs)
    - models/federated_noniid_history.json (training history)
"""

import sqlite3
import json
import os
import time
import argparse
import numpy as np
from datetime import datetime, timedelta
from collections import deque

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH      = os.path.join(BASE, "models", "fedshield_logs.db")
HISTORY_PATH = os.path.join(BASE, "models", "federated_noniid_history.json")
DRIFT_LOG    = os.path.join(BASE, "models", "drift_log.json")

# ── Thresholds ────────────────────────────────────────────────────────────────
DRIFT_CONFIG = {
    "false_positive_rate_threshold": 0.15,  # Alert if >15% of traffic misclassified
    "attack_spike_multiplier":       3.0,   # Alert if attack rate 3x baseline
    "window_minutes":                5,     # Rolling window for live metrics
    "min_samples":                   50,    # Minimum samples before alerting
}


# ── Live metrics from SQLite ──────────────────────────────────────────────────
def get_live_metrics(window_minutes: int = 5) -> dict:
    """Compute detection metrics from recent live capture data."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cutoff = (datetime.now() - timedelta(minutes=window_minutes)).strftime("%H:%M:%S")

        total   = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        attacks = conn.execute("SELECT COUNT(*) FROM detections WHERE tag='ATTACK'").fetchone()[0]
        blocked = conn.execute("SELECT COUNT(*) FROM detections WHERE blocked=1").fetchone()[0]

        # Recent window
        recent_total   = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE timestamp >= ?", (cutoff,)
        ).fetchone()[0]
        recent_attacks = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE tag='ATTACK' AND timestamp >= ?", (cutoff,)
        ).fetchone()[0]

        # Attack type distribution
        breakdown = conn.execute(
            "SELECT prediction, COUNT(*) as cnt FROM detections GROUP BY prediction"
        ).fetchall()

        # Confidence stats
        conf_stats = conn.execute(
            "SELECT AVG(confidence), MIN(confidence), MAX(confidence) FROM detections"
        ).fetchone()

        # Low confidence detections (potential drift signal)
        low_conf = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE confidence < 0.7"
        ).fetchone()[0]

        conn.close()

        attack_rate = attacks / max(total, 1)
        recent_rate = recent_attacks / max(recent_total, 1)
        low_conf_rate = low_conf / max(total, 1)

        return {
            "total":            total,
            "attacks":          attacks,
            "blocked":          blocked,
            "attack_rate":      round(attack_rate, 4),
            "recent_total":     recent_total,
            "recent_attacks":   recent_attacks,
            "recent_rate":      round(recent_rate, 4),
            "avg_confidence":   round(conf_stats[0] or 0, 4),
            "min_confidence":   round(conf_stats[1] or 0, 4),
            "low_conf_rate":    round(low_conf_rate, 4),
            "breakdown":        {r[0]: r[1] for r in breakdown},
        }
    except Exception as e:
        return {"error": str(e)}


# ── Training history analysis ─────────────────────────────────────────────────
def analyze_training_history() -> dict:
    """Analyze federated training history for convergence and stability."""
    try:
        with open(HISTORY_PATH) as f:
            history = json.load(f)

        f1_scores = [r["macro_f1"] for r in history]
        if len(f1_scores) < 3:
            return {"error": "Insufficient training history"}

        final_f1    = f1_scores[-1]
        best_f1     = max(f1_scores)
        worst_f1    = min(f1_scores)
        f1_std      = float(np.std(f1_scores))
        trend       = f1_scores[-1] - f1_scores[-3]  # Last 3 rounds trend

        # Convergence: std of last 3 rounds
        last3_std   = float(np.std(f1_scores[-3:]))
        converged   = last3_std < 0.005

        # Detect oscillation (sign changes in consecutive diffs)
        diffs       = [f1_scores[i+1] - f1_scores[i] for i in range(len(f1_scores)-1)]
        sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i]*diffs[i+1] < 0)
        oscillating  = sign_changes > len(diffs) * 0.4

        return {
            "rounds":        len(f1_scores),
            "final_f1":      round(final_f1, 4),
            "best_f1":       round(best_f1, 4),
            "worst_f1":      round(worst_f1, 4),
            "f1_std":        round(f1_std, 4),
            "trend":         round(trend, 4),
            "converged":     converged,
            "oscillating":   oscillating,
            "last3_std":     round(last3_std, 4),
            "sign_changes":  sign_changes,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Drift detection logic ─────────────────────────────────────────────────────
def detect_drift(metrics: dict, history: dict) -> list:
    """
    Run drift detection checks and return list of alerts.
    Each alert: {level, type, message, value, threshold}
    """
    alerts = []

    if "error" in metrics:
        return [{"level": "ERROR", "type": "DB_ERROR", "message": metrics["error"]}]

    # Check 1: Low confidence rate (model uncertainty — possible drift)
    if metrics["total"] > DRIFT_CONFIG["min_samples"]:
        if metrics["low_conf_rate"] > DRIFT_CONFIG["false_positive_rate_threshold"]:
            alerts.append({
                "level":     "WARNING",
                "type":      "LOW_CONFIDENCE",
                "message":   f"{metrics['low_conf_rate']*100:.1f}% of detections below 70% confidence",
                "value":     metrics["low_conf_rate"],
                "threshold": DRIFT_CONFIG["false_positive_rate_threshold"],
            })

    # Check 2: Attack rate spike
    if metrics["recent_total"] > 20:
        baseline = metrics["attack_rate"]
        recent   = metrics["recent_rate"]
        if baseline > 0 and recent > baseline * DRIFT_CONFIG["attack_spike_multiplier"]:
            alerts.append({
                "level":     "CRITICAL",
                "type":      "ATTACK_SPIKE",
                "message":   f"Attack rate spiked: {recent*100:.1f}% vs baseline {baseline*100:.1f}%",
                "value":     recent,
                "threshold": baseline * DRIFT_CONFIG["attack_spike_multiplier"],
            })

    # Check 3: Training convergence issues
    if "error" not in history:
        if history["oscillating"]:
            alerts.append({
                "level":   "WARNING",
                "type":    "OSCILLATION",
                "message": f"Training F1 oscillating ({history['sign_changes']} sign changes)",
                "value":   history["sign_changes"],
                "threshold": len([]) * 0.4,
            })

        if history["trend"] < -0.02:
            alerts.append({
                "level":   "WARNING",
                "type":    "F1_DECLINE",
                "message": f"F1 declining: {history['trend']:+.4f} over last 3 rounds",
                "value":   history["trend"],
                "threshold": -0.02,
            })

    # Check 4: Zero attacks in large sample (suspicious)
    if metrics["total"] > 1000 and metrics["attacks"] == 0:
        alerts.append({
            "level":   "INFO",
            "type":    "NO_ATTACKS",
            "message": f"No attacks in {metrics['total']:,} packets — model may be under-detecting",
            "value":   0,
            "threshold": 0,
        })

    return alerts


# ── Save drift log ────────────────────────────────────────────────────────────
def save_drift_log(metrics: dict, history: dict, alerts: list):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "metrics":   metrics,
        "history":   history,
        "alerts":    alerts,
        "status":    "DRIFT_DETECTED" if any(a["level"]=="CRITICAL" for a in alerts)
                     else "WARNING" if alerts else "HEALTHY",
    }
    try:
        existing = []
        if os.path.exists(DRIFT_LOG):
            with open(DRIFT_LOG) as f:
                existing = json.load(f)
        existing.append(log_entry)
        existing = existing[-100:]  # Keep last 100 entries
        with open(DRIFT_LOG, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save drift log: {e}")
    return log_entry


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(metrics: dict, history: dict, alerts: list):
    print("\n" + "="*60)
    print("FedShield — Model Drift Detection Report")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    print("\n📊 LIVE METRICS")
    if "error" not in metrics:
        print(f"  Total packets:      {metrics['total']:,}")
        print(f"  Attack rate:        {metrics['attack_rate']*100:.2f}%")
        print(f"  Recent rate:        {metrics['recent_rate']*100:.2f}% (last {DRIFT_CONFIG['window_minutes']}min)")
        print(f"  Avg confidence:     {metrics['avg_confidence']*100:.1f}%")
        print(f"  Low conf rate:      {metrics['low_conf_rate']*100:.1f}%")
        print(f"  Breakdown:          {metrics['breakdown']}")
    else:
        print(f"  Error: {metrics['error']}")

    print("\n📈 TRAINING HISTORY")
    if "error" not in history:
        print(f"  Rounds:             {history['rounds']}")
        print(f"  Final F1:           {history['final_f1']}")
        print(f"  Best F1:            {history['best_f1']}")
        print(f"  F1 std:             {history['f1_std']}")
        print(f"  Trend (last 3):     {history['trend']:+.4f}")
        print(f"  Converged:          {'✅ Yes' if history['converged'] else '⚠️ No'}")
        print(f"  Oscillating:        {'⚠️ Yes' if history['oscillating'] else '✅ No'}")
    else:
        print(f"  Error: {history['error']}")

    print("\n🚨 ALERTS")
    if not alerts:
        print("  ✅ No drift detected — model healthy")
    else:
        for a in alerts:
            icon = "🔴" if a["level"]=="CRITICAL" else "🟡" if a["level"]=="WARNING" else "🔵"
            print(f"  {icon} [{a['level']}] {a['type']}: {a['message']}")

    status = "🔴 DRIFT DETECTED" if any(a["level"]=="CRITICAL" for a in alerts) \
             else "🟡 WARNINGS" if alerts else "✅ HEALTHY"
    print(f"\n  Overall status: {status}")
    print("="*60)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedShield Drift Detection")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval in seconds")
    args = parser.parse_args()

    if args.watch:
        print(f"[Drift Monitor] Watching every {args.interval}s. Ctrl+C to stop.")
        try:
            while True:
                metrics = get_live_metrics(DRIFT_CONFIG["window_minutes"])
                history = analyze_training_history()
                alerts  = detect_drift(metrics, history)
                log     = save_drift_log(metrics, history, alerts)
                print_report(metrics, history, alerts)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[Drift Monitor] Stopped.")
    else:
        metrics = get_live_metrics(DRIFT_CONFIG["window_minutes"])
        history = analyze_training_history()
        alerts  = detect_drift(metrics, history)
        save_drift_log(metrics, history, alerts)
        print_report(metrics, history, alerts)