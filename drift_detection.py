"""FedShield — live model drift detection."""

import argparse
import json
import os
import sqlite3
import time

from datetime import datetime, timedelta

import numpy as np


BASE = os.path.dirname(
    os.path.abspath(__file__)
)

DB_PATH = os.path.join(
    BASE,
    "models",
    "fedshield_logs.db"
)

HISTORY_PATH = os.path.join(
    BASE,
    "models",
    "federated_noniid_history.json"
)

DRIFT_LOG = os.path.join(
    BASE,
    "models",
    "drift_log.json"
)


DRIFT_CONFIG = {
    "low_confidence_rate_threshold": 0.15,
    "attack_spike_multiplier": 3.0,
    "window_minutes": 5,
    "min_samples": 50,
}


def _table_exists(
    conn,
    table
):
    return (
        conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table'
            AND name=?
            """,
            (table,)
        ).fetchone()
        is not None
    )


def get_live_metrics(
    window_minutes=5
):
    """Get live detection statistics."""

    try:

        if not os.path.exists(DB_PATH):

            return {
                "error":
                    f"Database not found: {DB_PATH}"
            }

        conn = sqlite3.connect(
            DB_PATH,
            timeout=5
        )

        if not _table_exists(
            conn,
            "detections"
        ):

            conn.close()

            return {
                "error":
                    "detections table does not exist"
            }

        # New live_capture.py stores ISO timestamps.
        cutoff = (
            datetime.now()
            - timedelta(
                minutes=window_minutes
            )
        ).isoformat()

        total = conn.execute(
            "SELECT COUNT(*) FROM detections"
        ).fetchone()[0]

        attacks = conn.execute(
            """
            SELECT COUNT(*)
            FROM detections
            WHERE tag='ATTACK'
            """
        ).fetchone()[0]

        blocked = conn.execute(
            """
            SELECT COUNT(*)
            FROM detections
            WHERE blocked=1
            """
        ).fetchone()[0]

        recent_total = conn.execute(
            """
            SELECT COUNT(*)
            FROM detections
            WHERE timestamp >= ?
            """,
            (cutoff,)
        ).fetchone()[0]

        recent_attacks = conn.execute(
            """
            SELECT COUNT(*)
            FROM detections
            WHERE tag='ATTACK'
            AND timestamp >= ?
            """,
            (cutoff,)
        ).fetchone()[0]

        breakdown = conn.execute(
            """
            SELECT prediction, COUNT(*)
            FROM detections
            GROUP BY prediction
            """
        ).fetchall()

        conf = conn.execute(
            """
            SELECT
                AVG(confidence),
                MIN(confidence),
                MAX(confidence)
            FROM detections
            """
        ).fetchone()

        low_conf = conn.execute(
            """
            SELECT COUNT(*)
            FROM detections
            WHERE confidence < 0.7
            """
        ).fetchone()[0]

        conn.close()

        return {
            "total": total,
            "attacks": attacks,
            "blocked": blocked,

            "attack_rate": round(
                attacks / max(total, 1),
                4
            ),

            "recent_total": recent_total,
            "recent_attacks": recent_attacks,

            "recent_rate": round(
                recent_attacks
                / max(recent_total, 1),
                4
            ),

            "avg_confidence": round(
                float(conf[0] or 0),
                4
            ),

            "min_confidence": round(
                float(conf[1] or 0),
                4
            ),

            "low_conf_rate": round(
                low_conf
                / max(total, 1),
                4
            ),

            "breakdown": {
                str(k): v
                for k, v in breakdown
            },
        }

    except sqlite3.Error as e:

        return {
            "error": str(e)
        }


def analyze_training_history():
    """Analyze training F1 history."""

    try:

        with open(
            HISTORY_PATH,
            encoding="utf-8"
        ) as f:

            history = json.load(f)

        f1 = [
            float(r["macro_f1"])
            for r in history
            if "macro_f1" in r
        ]

        if len(f1) < 3:

            return {
                "error":
                    "Insufficient training history"
            }

        diffs = np.diff(f1)

        sign_changes = int(
            np.sum(
                diffs[:-1] * diffs[1:] < 0
            )
        ) if len(diffs) > 1 else 0

        last3_std = float(
            np.std(f1[-3:])
        )

        return {
            "rounds": len(f1),

            "final_f1": round(
                f1[-1],
                4
            ),

            "best_f1": round(
                max(f1),
                4
            ),

            "worst_f1": round(
                min(f1),
                4
            ),

            "f1_std": round(
                float(np.std(f1)),
                4
            ),

            "trend": round(
                f1[-1] - f1[-3],
                4
            ),

            "converged":
                last3_std < 0.005,

            "oscillating":
                sign_changes
                > max(
                    1,
                    int(len(diffs) * 0.4)
                ),

            "last3_std": round(
                last3_std,
                4
            ),

            "sign_changes":
                sign_changes,
        }

    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError
    ) as e:

        return {
            "error": str(e)
        }


def detect_drift(
    metrics,
    history
):
    """Run drift checks."""

    alerts = []

    if "error" in metrics:

        return [{
            "level": "ERROR",
            "type": "DB_ERROR",
            "message": metrics["error"]
        }]

    # Check 1: Low confidence
    if (
        metrics["total"]
        >= DRIFT_CONFIG["min_samples"]
        and
        metrics["low_conf_rate"]
        > DRIFT_CONFIG[
            "low_confidence_rate_threshold"
        ]
    ):

        alerts.append({
            "level": "WARNING",
            "type": "LOW_CONFIDENCE",

            "message":
                f"{metrics['low_conf_rate'] * 100:.1f}% "
                f"of detections are below "
                f"70% confidence",

            "value":
                metrics["low_conf_rate"],

            "threshold":
                DRIFT_CONFIG[
                    "low_confidence_rate_threshold"
                ],
        })

    # Check 2: Attack-rate spike
    if metrics["recent_total"] > 20:

        baseline = metrics["attack_rate"]
        recent = metrics["recent_rate"]

        if (
            baseline > 0
            and
            recent
            > baseline
            * DRIFT_CONFIG[
                "attack_spike_multiplier"
            ]
        ):

            alerts.append({
                "level": "CRITICAL",
                "type": "ATTACK_SPIKE",

                "message":
                    f"Attack rate spiked to "
                    f"{recent * 100:.1f}% "
                    f"vs baseline "
                    f"{baseline * 100:.1f}%",

                "value": recent,

                "threshold":
                    baseline
                    * DRIFT_CONFIG[
                        "attack_spike_multiplier"
                    ],
            })

    # Check 3: Training instability
    if "error" not in history:

        if history["oscillating"]:

            threshold = max(
                1,
                int(
                    max(
                        history["rounds"] - 2,
                        1
                    ) * 0.4
                )
            )

            alerts.append({
                "level": "WARNING",
                "type": "OSCILLATION",

                "message":
                    "Training F1 is oscillating "
                    f"({history['sign_changes']} "
                    "sign changes)",

                "value":
                    history["sign_changes"],

                "threshold":
                    threshold,
            })

        if history["trend"] < -0.02:

            alerts.append({
                "level": "WARNING",
                "type": "F1_DECLINE",

                "message":
                    f"F1 declined by "
                    f"{history['trend']:+.4f} "
                    "over the last 3 rounds",

                "value":
                    history["trend"],

                "threshold":
                    -0.02,
            })

    # Check 4: No attacks
    if (
        metrics["total"] > 1000
        and
        metrics["attacks"] == 0
    ):

        alerts.append({
            "level": "INFO",
            "type": "NO_ATTACKS",

            "message":
                f"No attacks in "
                f"{metrics['total']:,} logged packets "
                "— verify that the detector is receiving "
                "representative traffic",

            "value": 0,
            "threshold": 0,
        })

    return alerts


def save_drift_log(
    metrics,
    history,
    alerts
):
    """Persist the latest drift analysis."""

    entry = {
        "timestamp":
            datetime.utcnow().isoformat(),

        "metrics":
            metrics,

        "history":
            history,

        "alerts":
            alerts,

        "status":
            (
                "DRIFT_DETECTED"
                if any(
                    a["level"] == "CRITICAL"
                    for a in alerts
                )
                else
                "WARNING"
                if alerts
                else
                "HEALTHY"
            )
    }

    try:

        os.makedirs(
            os.path.dirname(DRIFT_LOG),
            exist_ok=True
        )

        existing = []

        if os.path.exists(DRIFT_LOG):

            with open(
                DRIFT_LOG,
                encoding="utf-8"
            ) as f:

                existing = json.load(f)

        if not isinstance(
            existing,
            list
        ):

            existing = []

        existing.append(entry)

        with open(
            DRIFT_LOG,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                existing[-100:],
                f,
                indent=2
            )

    except (
        OSError,
        ValueError,
        json.JSONDecodeError
    ) as e:

        print(
            f"[WARN] Could not save drift log: {e}"
        )

    return entry


def print_report(
    metrics,
    history,
    alerts
):

    print("\n" + "=" * 60)
    print(
        "FedShield — Model Drift Detection Report"
    )

    print(
        f"Timestamp: "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    )

    print("=" * 60)

    print("\nLIVE METRICS")

    if "error" in metrics:

        print(
            f"  Error: {metrics['error']}"
        )

    else:

        print(
            f"  Total packets:  "
            f"{metrics['total']:,}"
        )

        print(
            f"  Attack rate:    "
            f"{metrics['attack_rate'] * 100:.2f}%"
        )

        print(
            f"  Recent rate:    "
            f"{metrics['recent_rate'] * 100:.2f}%"
        )

        print(
            f"  Avg confidence: "
            f"{metrics['avg_confidence'] * 100:.1f}%"
        )

        print(
            f"  Low conf rate:  "
            f"{metrics['low_conf_rate'] * 100:.1f}%"
        )

        print(
            f"  Breakdown:      "
            f"{metrics['breakdown']}"
        )

    print("\nTRAINING HISTORY")

    if "error" in history:

        print(
            f"  Error: {history['error']}"
        )

    else:

        print(
            f"  Rounds: {history['rounds']} | "
            f"Final F1: {history['final_f1']} | "
            f"Best F1: {history['best_f1']}"
        )

        print(
            f"  Trend: {history['trend']:+.4f} | "
            f"Converged: {history['converged']} | "
            f"Oscillating: {history['oscillating']}"
        )

    print("\nALERTS")

    if not alerts:

        print(
            "  No drift detected — model healthy"
        )

    for alert in alerts:

        print(
            f"  [{alert['level']}] "
            f"{alert['type']}: "
            f"{alert['message']}"
        )

    print("=" * 60)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--watch",
        action="store_true"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60
    )

    args = parser.parse_args()

    if args.interval < 1:

        parser.error(
            "--interval must be >= 1"
        )

    while True:

        metrics = get_live_metrics(
            DRIFT_CONFIG["window_minutes"]
        )

        history = analyze_training_history()

        alerts = detect_drift(
            metrics,
            history
        )

        save_drift_log(
            metrics,
            history,
            alerts
        )

        print_report(
            metrics,
            history,
            alerts
        )

        if not args.watch:
            break

        try:

            time.sleep(
                args.interval
            )

        except KeyboardInterrupt:

            print(
                "\n[Drift Monitor] Stopped."
            )

            break


if __name__ == "__main__":
    main()