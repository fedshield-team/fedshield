"""FedShield — AbuseIPDB threat-intelligence enrichment."""

import ipaddress
import json
import os
import sqlite3
import time

from datetime import datetime
from typing import Any

import requests


BASE = os.path.dirname(
    os.path.abspath(__file__)
)

DB_PATH = os.path.join(
    BASE,
    "models",
    "fedshield_logs.db"
)

CACHE_FILE = os.path.join(
    BASE,
    "models",
    "threat_intel_cache.json"
)

API_URL = (
    "https://api.abuseipdb.com/api/v2/check"
)

MAX_AGE = 90

CACHE_TTL_HOURS = 24

REQUEST_DELAY_SECONDS = 0.5


def get_api_key():

    return os.getenv(
        "ABUSEIPDB_API_KEY"
    )


def load_cache():

    try:

        with open(
            CACHE_FILE,
            encoding="utf-8"
        ) as f:

            value = json.load(f)

        return (
            value
            if isinstance(value, dict)
            else {}
        )

    except (
        OSError,
        json.JSONDecodeError
    ):

        return {}


def save_cache(cache):

    try:

        os.makedirs(
            os.path.dirname(
                CACHE_FILE
            ),
            exist_ok=True
        )

        tmp = (
            CACHE_FILE
            + ".tmp"
        )

        with open(
            tmp,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                cache,
                f,
                indent=2
            )

        os.replace(
            tmp,
            CACHE_FILE
        )

    except OSError as e:

        print(
            f"[ThreatIntel] "
            f"Cache write failed: {e}"
        )


def is_private_or_local_ip(
    ip
):

    try:

        addr = ipaddress.ip_address(
            ip.strip()
        )

        return (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_unspecified
            or addr.is_multicast
        )

    except ValueError:

        return False


def _base_result(
    ip,
    **kwargs
):

    result = {
        "ip": ip,
        "abuse_score": -1,
        "total_reports": -1,
        "country": "Unknown",
        "isp": "Unknown",
        "domain": "",
        "is_tor": False,
        "is_known_attacker": False,
        "last_reported": None,
        "cached": False
    }

    result.update(
        kwargs
    )

    return result


def check_ip(
    ip: str
) -> dict[str, Any]:

    ip = ip.strip()

    try:

        ipaddress.ip_address(
            ip
        )

    except ValueError:

        return _base_result(
            ip,
            error="Invalid IP address"
        )

    if is_private_or_local_ip(
        ip
    ):

        return _base_result(
            ip,
            abuse_score=0,
            total_reports=0,
            country="LOCAL",
            isp="Private/Local Network",
            note="Private/local IP — not checked"
        )

    api_key = get_api_key()

    if not api_key:

        return _base_result(
            ip,
            error=(
                "ABUSEIPDB_API_KEY "
                "is not configured"
            )
        )

    cache = load_cache()

    entry = cache.get(
        ip
    )

    if entry:

        try:

            age_hours = (
                time.time()
                - float(
                    entry.get(
                        "cached_at",
                        0
                    )
                )
            ) / 3600

            if age_hours < CACHE_TTL_HOURS:

                result = dict(
                    entry
                )

                result["cached"] = True

                return result

        except (
            TypeError,
            ValueError
        ):

            pass

    try:

        response = requests.get(
            API_URL,
            headers={
                "Key": api_key,
                "Accept": "application/json"
            },
            params={
                "ipAddress": ip,
                "maxAgeInDays": MAX_AGE
            },
            timeout=8
        )

        response.raise_for_status()

        data = response.json().get(
            "data",
            {}
        )

        score = int(
            data.get(
                "abuseConfidenceScore",
                0
            )
            or 0
        )

        total_reports = int(
            data.get(
                "totalReports",
                0
            )
            or 0
        )

        result = _base_result(
            ip,

            abuse_score=score,

            total_reports=total_reports,

            country=data.get(
                "countryCode",
                "Unknown"
            ),

            isp=data.get(
                "isp",
                "Unknown"
            ),

            domain=data.get(
                "domain",
                ""
            ),

            is_tor=bool(
                data.get(
                    "isTor",
                    False
                )
            ),

            is_known_attacker=(
                score >= 25
            ),

            last_reported=data.get(
                "lastReportedAt"
            ),

            cached=False,

            cached_at=time.time()
        )

        cache[ip] = result

        save_cache(
            cache
        )

        return result

    except requests.Timeout:

        return _base_result(
            ip,
            error=(
                "AbuseIPDB request timed out"
            )
        )

    except requests.HTTPError as e:

        return _base_result(
            ip,
            error=(
                f"AbuseIPDB HTTP error: {e}"
            )
        )

    except requests.RequestException as e:

        return _base_result(
            ip,
            error=(
                f"AbuseIPDB request failed: {e}"
            )
        )

    except (
        ValueError,
        KeyError,
        TypeError
    ) as e:

        return _base_result(
            ip,
            error=(
                f"Invalid AbuseIPDB response: {e}"
            )
        )


def enrich_blocked_ips():

    try:

        with sqlite3.connect(
            DB_PATH
        ) as conn:

            rows = conn.execute(
                """
                SELECT DISTINCT src
                FROM detections
                WHERE blocked=1
                AND src IS NOT NULL
                AND src != ''
                """
            ).fetchall()

    except sqlite3.Error as e:

        print(
            f"[ThreatIntel] "
            f"Database error: {e}"
        )

        return []

    ips = [
        row[0]
        for row in rows
    ]

    if not ips:

        print(
            "[ThreatIntel] "
            "No blocked IPs found."
        )

        return []

    if not get_api_key():

        print(
            "[ThreatIntel] "
            "ABUSEIPDB_API_KEY "
            "is not configured."
        )

        return []

    results = []

    for i, ip in enumerate(
        ips
    ):

        result = check_ip(
            ip
        )

        results.append(
            result
        )

        summary = (
            result.get("note")
            or result.get("error")
            or (
                f"Score "
                f"{result['abuse_score']}/100 | "
                f"Reports "
                f"{result['total_reports']} | "
                f"{result['country']}"
            )
        )

        print(
            f"  {ip:40} -> "
            f"{summary}"
        )

        if i < len(ips) - 1:

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    return results


def print_threat_report(
    results
):

    print(
        "\n" + "=" * 60
    )

    print(
        "FedShield — "
        "Threat Intelligence Report"
    )

    print(
        f"Timestamp: "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    )

    print(
        "=" * 60
    )

    known = sum(
        bool(
            r.get(
                "is_known_attacker"
            )
        )
        for r in results
    )

    tor = sum(
        bool(
            r.get("is_tor")
        )
        for r in results
    )

    errors = sum(
        bool(
            r.get("error")
        )
        for r in results
    )

    print(
        f"Total: {len(results)} | "
        f"Known attackers: {known} | "
        f"TOR: {tor} | "
        f"Errors: {errors}"
    )

    for result in results:

        if (
            result.get(
                "is_known_attacker"
            )
            or
            result.get(
                "is_tor"
            )
        ):

            print(
                f"  {result['ip']} | "
                f"score={result['abuse_score']} | "
                f"reports={result['total_reports']} | "
                f"{result['country']} | "
                f"{result['isp']}"
            )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    results = enrich_blocked_ips()

    if results:

        print_threat_report(
            results
        )