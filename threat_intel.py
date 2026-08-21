"""
FedShield — AbuseIPDB Threat Intelligence Integration

Cross-references blocked IPs against the AbuseIPDB threat database.
Automatically called when live_capture.py blocks an IP.

Usage:
    from threat_intel import check_ip, enrich_blocked_ips

    result = check_ip("8.8.8.8")

Configuration:
    ABUSEIPDB_API_KEY must be provided through the environment.

The API key is intentionally NOT stored in source code.

Example .env:
    ABUSEIPDB_API_KEY=your-abuseipdb-api-key
"""

import json
import os
import sqlite3
import time
from datetime import datetime
from typing import Any

import requests


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(
    BASE,
    "models",
    "fedshield_logs.db",
)

CACHE_FILE = os.path.join(
    BASE,
    "models",
    "threat_intel_cache.json",
)

API_URL = "https://api.abuseipdb.com/api/v2/check"

# AbuseIPDB looks this many days back when calculating threat reputation.
MAX_AGE = 90

# Cache results for 24 hours to reduce API calls.
CACHE_TTL_HOURS = 24

# Small delay between requests to be respectful of API rate limits.
REQUEST_DELAY_SECONDS = 0.5


def get_api_key() -> str | None:
    """
    Read the AbuseIPDB API key from the environment.

    Returns:
        The API key if configured, otherwise None.

    We intentionally do not raise an exception here because threat
    intelligence is an optional enrichment layer. FedShield's core
    detection pipeline should still be able to run without it.
    """
    return os.getenv("ABUSEIPDB_API_KEY")


# ──────────────────────────────────────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────────────────────────────────────

def load_cache() -> dict:
    """Load the local threat-intelligence cache."""
    try:
        if not os.path.exists(CACHE_FILE):
            return {}

        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data if isinstance(data, dict) else {}

    except (OSError, json.JSONDecodeError):
        return {}


def save_cache(cache: dict) -> None:
    """Persist the local threat-intelligence cache."""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    except OSError as e:
        print(f"[ThreatIntel] Cache write failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# IP helpers
# ──────────────────────────────────────────────────────────────────────────────

def is_private_or_local_ip(ip: str) -> bool:
    """
    Detect common private/local IPv4 addresses.

    AbuseIPDB cannot provide meaningful global reputation information
    for private addresses, so these are skipped.
    """
    private_prefixes = (
        "192.168.",
        "10.",
        "172.16.",
        "172.17.",
        "172.18.",
        "172.19.",
        "172.20.",
        "172.21.",
        "172.22.",
        "172.23.",
        "172.24.",
        "172.25.",
        "172.26.",
        "172.27.",
        "172.28.",
        "172.29.",
        "172.30.",
        "172.31.",
        "127.",
        "0.",
    )

    return any(ip.startswith(prefix) for prefix in private_prefixes)


# ──────────────────────────────────────────────────────────────────────────────
# AbuseIPDB lookup
# ──────────────────────────────────────────────────────────────────────────────

def check_ip(ip: str) -> dict[str, Any]:
    """
    Query AbuseIPDB for threat intelligence on an IP address.

    Results are cached for 24 hours to reduce unnecessary API calls.

    Returns:
        {
            ip,
            abuse_score,
            total_reports,
            country,
            isp,
            domain,
            is_tor,
            is_known_attacker,
            last_reported,
            cached
        }
    """

    ip = ip.strip()

    # ── Validate input ────────────────────────────────────────────────────────
    if not ip:
        return {
            "ip": ip,
            "abuse_score": -1,
            "total_reports": -1,
            "country": "Unknown",
            "isp": "Unknown",
            "domain": "",
            "is_tor": False,
            "is_known_attacker": False,
            "last_reported": None,
            "cached": False,
            "error": "IP address is empty",
        }

    # ── Skip private/local addresses ──────────────────────────────────────────
    if is_private_or_local_ip(ip):
        return {
            "ip": ip,
            "abuse_score": 0,
            "total_reports": 0,
            "country": "LOCAL",
            "isp": "Private Network",
            "domain": "",
            "is_tor": False,
            "is_known_attacker": False,
            "last_reported": None,
            "cached": False,
            "note": "Private/local IP — not checked",
        }

    # ── Read API key ──────────────────────────────────────────────────────────
    api_key = get_api_key()

    if not api_key:
        return {
            "ip": ip,
            "abuse_score": -1,
            "total_reports": -1,
            "country": "Unknown",
            "isp": "Unknown",
            "domain": "",
            "is_tor": False,
            "is_known_attacker": False,
            "last_reported": None,
            "cached": False,
            "error": "ABUSEIPDB_API_KEY is not configured",
        }

    # ── Check cache ───────────────────────────────────────────────────────────
    cache = load_cache()

    if ip in cache:
        entry = cache[ip]

        cached_at = entry.get("cached_at", 0)

        try:
            age_hours = (
                time.time() - float(cached_at)
            ) / 3600
        except (TypeError, ValueError):
            age_hours = CACHE_TTL_HOURS + 1

        if age_hours < CACHE_TTL_HOURS:
            cached_result = dict(entry)
            cached_result["cached"] = True
            return cached_result

    # ── Query AbuseIPDB ───────────────────────────────────────────────────────
    try:
        response = requests.get(
            API_URL,
            headers={
                "Key": api_key,
                "Accept": "application/json",
            },
            params={
                "ipAddress": ip,
                "maxAgeInDays": MAX_AGE,
            },
            timeout=8,
        )

        response.raise_for_status()

        payload = response.json()
        data = payload.get("data", {})

        abuse_score = data.get(
            "abuseConfidenceScore",
            0,
        )

        result = {
            "ip": ip,
            "abuse_score": abuse_score,
            "total_reports": data.get(
                "totalReports",
                0,
            ),
            "country": data.get(
                "countryCode",
                "Unknown",
            ),
            "isp": data.get(
                "isp",
                "Unknown",
            ),
            "domain": data.get(
                "domain",
                "",
            ),
            "is_tor": data.get(
                "isTor",
                False,
            ),
            "is_known_attacker": abuse_score >= 25,
            "last_reported": data.get(
                "lastReportedAt",
                None,
            ),
            "cached": False,
            "cached_at": time.time(),
        }

        # Save successful lookup.
        cache[ip] = result
        save_cache(cache)

        return result

    except requests.Timeout:
        return {
            "ip": ip,
            "abuse_score": -1,
            "total_reports": -1,
            "country": "Unknown",
            "isp": "Unknown",
            "domain": "",
            "is_tor": False,
            "is_known_attacker": False,
            "last_reported": None,
            "cached": False,
            "error": "AbuseIPDB request timed out",
        }

    except requests.HTTPError as e:
        return {
            "ip": ip,
            "abuse_score": -1,
            "total_reports": -1,
            "country": "Unknown",
            "isp": "Unknown",
            "domain": "",
            "is_tor": False,
            "is_known_attacker": False,
            "last_reported": None,
            "cached": False,
            "error": f"AbuseIPDB HTTP error: {e}",
        }

    except requests.RequestException as e:
        return {
            "ip": ip,
            "abuse_score": -1,
            "total_reports": -1,
            "country": "Unknown",
            "isp": "Unknown",
            "domain": "",
            "is_tor": False,
            "is_known_attacker": False,
            "last_reported": None,
            "cached": False,
            "error": f"AbuseIPDB request failed: {e}",
        }

    except (ValueError, KeyError, TypeError) as e:
        return {
            "ip": ip,
            "abuse_score": -1,
            "total_reports": -1,
            "country": "Unknown",
            "isp": "Unknown",
            "domain": "",
            "is_tor": False,
            "is_known_attacker": False,
            "last_reported": None,
            "cached": False,
            "error": f"Invalid AbuseIPDB response: {e}",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Enrichment
# ──────────────────────────────────────────────────────────────────────────────

def enrich_blocked_ips() -> list[dict[str, Any]]:
    """
    Pull all blocked source IPs from SQLite and enrich them with
    AbuseIPDB threat intelligence.

    Returns:
        List of threat-intelligence results.
    """

    try:
        conn = sqlite3.connect(DB_PATH)

        rows = conn.execute(
            """
            SELECT DISTINCT src
            FROM detections
            WHERE blocked = 1
              AND src IS NOT NULL
              AND src != ''
            """
        ).fetchall()

        conn.close()

    except sqlite3.Error as e:
        print(f"[ThreatIntel] Database error: {e}")
        return []

    ips = [row[0] for row in rows]

    if not ips:
        print(
            "[ThreatIntel] No blocked IPs found in database."
        )
        return []

    if not get_api_key():
        print(
            "[ThreatIntel] ABUSEIPDB_API_KEY is not configured. "
            "Skipping external threat intelligence."
        )
        return []

    print(
        f"\n[ThreatIntel] Checking "
        f"{len(ips)} blocked IP(s) against AbuseIPDB...\n"
    )

    results = []

    for index, ip in enumerate(ips):
        result = check_ip(ip)
        results.append(result)

        score = result.get("abuse_score", 0)
        country = result.get("country", "?")
        isp = result.get("isp", "?")
        reports = result.get("total_reports", 0)

        known = (
            "⚠️ KNOWN ATTACKER"
            if result.get("is_known_attacker")
            else "✅ Clean"
        )

        tor = (
            " | 🧅 TOR EXIT NODE"
            if result.get("is_tor")
            else ""
        )

        note = result.get("note", "")
        error = result.get("error", "")

        if note:
            print(
                f"  {ip:20} → {note}"
            )

        elif error:
            print(
                f"  {ip:20} → ERROR: {error}"
            )

        else:
            print(
                f"  {ip:20} → "
                f"Score: {score:3}/100 | "
                f"Reports: {reports:4} | "
                f"{country} | "
                f"{isp[:30]} | "
                f"{known}{tor}"
            )

        # Avoid unnecessary delay after the final request.
        if index < len(ips) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Human-readable report
# ──────────────────────────────────────────────────────────────────────────────

def print_threat_report(results: list[dict[str, Any]]) -> None:
    """Print a summary of threat-intelligence results."""

    print("\n" + "=" * 60)
    print("FedShield — Threat Intelligence Report")
    print(
        f"Timestamp: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("=" * 60)

    known_attackers = [
        result
        for result in results
        if result.get("is_known_attacker")
    ]

    tor_nodes = [
        result
        for result in results
        if result.get("is_tor")
    ]

    errors = [
        result
        for result in results
        if result.get("error")
    ]

    clean = [
        result
        for result in results
        if (
            not result.get("is_known_attacker")
            and not result.get("is_tor")
            and not result.get("error")
        )
    ]

    print(
        f"\n  Total IPs checked:    {len(results)}"
    )
    print(
        f"  Known attackers:      {len(known_attackers)} 🔴"
    )
    print(
        f"  TOR exit nodes:       {len(tor_nodes)} 🧅"
    )
    print(
        f"  Clean / local:        {len(clean)} ✅"
    )
    print(
        f"  Lookup errors:        {len(errors)} ⚠️"
    )

    if known_attackers:
        print("\n  🔴 Known Malicious IPs:")

        for result in known_attackers:
            print(
                f"    {result['ip']:20} "
                f"Score:{result['abuse_score']:3}/100 "
                f"Reports:{result['total_reports']:4} "
                f"{result['country']} — "
                f"{result['isp']}"
            )

    if tor_nodes:
        print("\n  🧅 TOR Exit Nodes:")

        for result in tor_nodes:
            print(
                f"    {result['ip']:20} "
                f"{result['country']} — "
                f"{result['isp']}"
            )

    if errors:
        print("\n  ⚠️ Lookup Errors:")

        for result in errors:
            print(
                f"    {result['ip']:20} "
                f"{result.get('error', 'Unknown error')}"
            )

    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Manual test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = enrich_blocked_ips()

    if results:
        print_threat_report(results)

    elif get_api_key():
        # Manual test using a public IP.
        #
        # This does NOT mean the IP is malicious; it simply demonstrates
        # that the AbuseIPDB lookup integration is reachable.
        print(
            "\n[Demo] No blocked IPs in DB — "
            "testing AbuseIPDB integration..."
        )

        result = check_ip("8.8.8.8")

        print(f"\n  IP:            {result['ip']}")
        print(
            f"  Abuse Score:   "
            f"{result['abuse_score']}/100"
        )
        print(
            f"  Total Reports: "
            f"{result['total_reports']}"
        )
        print(
            f"  Country:       "
            f"{result['country']}"
        )
        print(
            f"  ISP:            "
            f"{result['isp']}"
        )
        print(
            f"  Known Attack:  "
            f"{result['is_known_attacker']}"
        )
        print(
            f"  TOR Node:      "
            f"{result['is_tor']}"
        )

        print("\n✅ AbuseIPDB integration working!")

    else:
        print(
            "\n[ThreatIntel] AbuseIPDB integration is disabled."
        )
        print(
            "Set ABUSEIPDB_API_KEY in your environment "
            "to enable threat intelligence."
        )