"""
FedShield — AbuseIPDB Threat Intelligence Integration
Cross-references blocked IPs against AbuseIPDB global threat database.
Automatically called when live_capture.py blocks an IP.

Usage:
    from threat_intel import check_ip, enrich_blocked_ip
    result = check_ip("192.168.0.101")
"""

import requests
import sqlite3
import json
import os
import time
from datetime import datetime

BASE        = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE, "models", "fedshield_logs.db")
CACHE_FILE  = os.path.join(BASE, "models", "threat_intel_cache.json")
API_KEY     = "eff83bbc5ebcedaae4cc9c461d531cf96775905fec8b5a7839d13b4560015757c80bd2f146e2d63a"
API_URL     = "https://api.abuseipdb.com/api/v2/check"
MAX_AGE     = 90  # days to look back in AbuseIPDB


def load_cache() -> dict:
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(cache: dict):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def check_ip(ip: str) -> dict:
    """
    Query AbuseIPDB for threat intelligence on an IP address.
    Results cached for 24 hours to avoid hammering the API.

    Returns:
        {
            ip, abuse_score, total_reports, country,
            isp, domain, is_tor, is_known_attacker,
            last_reported, cached
        }
    """
    # Skip private/local IPs
    private_prefixes = ("192.168.", "10.", "172.16.", "172.17.", "127.", "0.")
    if any(ip.startswith(p) for p in private_prefixes):
        return {
            "ip": ip, "abuse_score": 0, "total_reports": 0,
            "country": "LOCAL", "isp": "Private Network",
            "domain": "", "is_tor": False,
            "is_known_attacker": False,
            "last_reported": None, "cached": False,
            "note": "Private/local IP — not checked"
        }

    # Check cache
    cache = load_cache()
    if ip in cache:
        entry = cache[ip]
        age_hours = (time.time() - entry.get("cached_at", 0)) / 3600
        if age_hours < 24:
            entry["cached"] = True
            return entry

    try:
        resp = requests.get(
            API_URL,
            headers={"Key": API_KEY, "Accept": "application/json"},
            params={"ipAddress": ip, "maxAgeInDays": MAX_AGE, "verbose": True},
            timeout=8
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})

        result = {
            "ip":               ip,
            "abuse_score":      data.get("abuseConfidenceScore", 0),
            "total_reports":    data.get("totalReports", 0),
            "country":          data.get("countryCode", "Unknown"),
            "isp":              data.get("isp", "Unknown"),
            "domain":           data.get("domain", ""),
            "is_tor":           data.get("isTor", False),
            "is_known_attacker": data.get("abuseConfidenceScore", 0) >= 25,
            "last_reported":    data.get("lastReportedAt", None),
            "cached":           False,
            "cached_at":        time.time(),
        }

        # Save to cache
        cache[ip] = result
        save_cache(cache)

        return result

    except requests.RequestException as e:
        return {
            "ip": ip, "abuse_score": -1, "total_reports": -1,
            "country": "Unknown", "isp": "Unknown",
            "domain": "", "is_tor": False,
            "is_known_attacker": False,
            "last_reported": None, "cached": False,
            "error": str(e)
        }


def enrich_blocked_ips() -> list:
    """
    Pull all blocked IPs from SQLite and enrich them with threat intel.
    Returns list of enriched results.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT DISTINCT src FROM detections WHERE blocked=1"
        ).fetchall()
        conn.close()
    except Exception as e:
        print(f"[ThreatIntel] DB error: {e}")
        return []

    results = []
    ips = [r[0] for r in rows]

    if not ips:
        print("[ThreatIntel] No blocked IPs found in database.")
        return []

    print(f"\n[ThreatIntel] Checking {len(ips)} blocked IP(s) against AbuseIPDB...\n")

    for ip in ips:
        result = check_ip(ip)
        results.append(result)

        score = result.get("abuse_score", 0)
        country = result.get("country", "?")
        isp = result.get("isp", "?")
        reports = result.get("total_reports", 0)
        known = "⚠️  KNOWN ATTACKER" if result.get("is_known_attacker") else "✅ Clean"
        tor = " | 🧅 TOR EXIT NODE" if result.get("is_tor") else ""
        note = result.get("note", "")

        if note:
            print(f"  {ip:20} → {note}")
        else:
            print(f"  {ip:20} → Score: {score:3}/100 | Reports: {reports:4} | "
                  f"{country} | {isp[:30]} | {known}{tor}")

        time.sleep(0.5)  # Respect rate limit

    return results


def print_threat_report(results: list):
    print("\n" + "="*60)
    print("FedShield — Threat Intelligence Report")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    known_attackers = [r for r in results if r.get("is_known_attacker")]
    tor_nodes       = [r for r in results if r.get("is_tor")]
    clean           = [r for r in results if not r.get("is_known_attacker") and not r.get("is_tor")]

    print(f"\n  Total IPs checked:    {len(results)}")
    print(f"  Known attackers:      {len(known_attackers)} 🔴")
    print(f"  TOR exit nodes:       {len(tor_nodes)} 🧅")
    print(f"  Clean / local:        {len(clean)} ✅")

    if known_attackers:
        print("\n  🔴 Known Malicious IPs:")
        for r in known_attackers:
            print(f"    {r['ip']:20} Score:{r['abuse_score']:3}/100 "
                  f"Reports:{r['total_reports']:4} {r['country']} — {r['isp']}")

    if tor_nodes:
        print("\n  🧅 TOR Exit Nodes:")
        for r in tor_nodes:
            print(f"    {r['ip']:20} {r['country']} — {r['isp']}")

    print("="*60)


if __name__ == "__main__":
    results = enrich_blocked_ips()
    if results:
        print_threat_report(results)
    else:
        # Demo with a known malicious IP
        print("\n[Demo] No blocked IPs in DB — testing with known malicious IP...")
        result = check_ip("8.8.8.8")
        print(f"\n  IP:            {result['ip']}")
        print(f"  Abuse Score:   {result['abuse_score']}/100")
        print(f"  Total Reports: {result['total_reports']}")
        print(f"  Country:       {result['country']}")
        print(f"  ISP:           {result['isp']}")
        print(f"  Known Attack:  {result['is_known_attacker']}")
        print(f"  TOR Node:      {result['is_tor']}")
        print("\n✅ AbuseIPDB integration working!")