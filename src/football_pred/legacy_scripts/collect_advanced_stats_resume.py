"""
Resume collection of advanced match statistics from API-Football.
Prioritizes seasons closest to the test set (2024, 2023, then older).
Uses existing cache to skip already collected matches.
Saves cache every 50 matches. Stops gracefully when quota runs out.
"""

import json
import os
import sys
import time

import pandas as pd
import requests

API_KEY = os.environ.get("FOOTBALL_API_SPORTS", "")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

CACHE_FILE = "/home/ubuntu/football_pred/dataset/advanced_stats_cache.json"
DATASET_FILE = "/home/ubuntu/football_pred/dataset/football_matches.csv"
LOG_FILE = "/tmp/collect_resume.log"

# Priority order: most recent uncollected seasons first
PRIORITY_SEASONS = [2024, 2023, 2022, 2021, 2020, 2019]

STAT_KEYS = [
    "Shots on Goal", "Total Shots", "Shots insidebox", "Shots outsidebox",
    "Ball Possession", "Total passes", "Passes accurate", "Passes %",
    "Corner Kicks", "Fouls", "expected_goals",
]

QUOTA_BUFFER = 10  # Stop when this many requests remain


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def check_remaining():
    """Check how many API requests remain today."""
    try:
        r = requests.get(f"{BASE_URL}/status", headers=HEADERS, timeout=10)
        data = r.json()["response"]
        current = data["requests"]["current"]
        limit = data["requests"]["limit_day"]
        return limit - current
    except Exception as e:
        log(f"  Warning: could not check quota: {e}")
        return None


def fetch_stats(fixture_id):
    """Fetch statistics for a single fixture."""
    r = requests.get(
        f"{BASE_URL}/fixtures/statistics",
        headers=HEADERS,
        params={"fixture": fixture_id},
        timeout=15,
    )
    data = r.json()
    if data.get("results", 0) == 0:
        return None

    result = {}
    for team_data in data["response"]:
        team_id = str(team_data["team"]["id"])
        team_name = team_data["team"]["name"]
        stats = {"team_name": team_name}
        for s in team_data.get("statistics", []):
            if s["type"] in STAT_KEYS:
                stats[s["type"]] = s["value"]
        # Also check for xG
        for s in team_data.get("statistics", []):
            if "expected" in s["type"].lower() and "goal" in s["type"].lower():
                stats["expected_goals"] = s["value"]
        result[team_id] = stats
    return result if result else None


def main():
    # Load cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
    else:
        cache = {}
    log(f"Cache loaded: {len(cache)} entries")

    # Load dataset
    df = pd.read_csv(DATASET_FILE)
    df["date"] = pd.to_datetime(df["date"])
    log(f"Dataset: {len(df)} matches")

    # Check initial quota
    remaining = check_remaining()
    log(f"API requests remaining: {remaining}")
    if remaining is not None and remaining <= QUOTA_BUFFER:
        log("Quota exhausted. Stopping.")
        return

    # Build list of fixture IDs to collect, prioritized by season
    to_collect = []
    for season in PRIORITY_SEASONS:
        season_df = df[df["season"] == season]
        season_ids = [
            str(int(fid))
            for fid in season_df["fixture_id"]
            if str(int(fid)) not in cache
        ]
        to_collect.extend([(fid, season) for fid in season_ids])
        cached_count = len(season_df) - len(season_ids)
        log(f"  Season {season}: {len(season_ids)} to collect ({cached_count} already cached)")

    log(f"Total to collect: {len(to_collect)}")
    if not to_collect:
        log("Nothing to collect!")
        return

    # Collect
    collected = 0
    errors = 0
    start = time.time()
    requests_used = 0

    for i, (fid, season) in enumerate(to_collect):
        # Check quota periodically
        if requests_used > 0 and requests_used % 50 == 0:
            remaining = check_remaining()
            requests_used += 1  # status check counts as a request
            if remaining is not None and remaining <= QUOTA_BUFFER:
                log(f"Quota nearly exhausted ({remaining} remaining). Stopping gracefully.")
                break

        try:
            stats = fetch_stats(fid)
            requests_used += 1
            if stats:
                cache[fid] = stats
                collected += 1
            else:
                cache[fid] = {}  # Mark as attempted (no stats available)
                collected += 1
        except Exception as e:
            errors += 1
            log(f"  Error on fixture {fid}: {e}")

        # Save cache every 50 matches
        if collected > 0 and collected % 50 == 0:
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)
            elapsed = time.time() - start
            rate = collected / elapsed * 60 if elapsed > 0 else 0
            log(f"Progress: {collected}/{len(to_collect)} collected, {errors} errors, "
                f"{rate:.0f} req/min, cache={len(cache)} entries")

        # Rate limiting: ~250 req/min to stay safe
        time.sleep(0.25)

    # Final save
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

    elapsed = time.time() - start
    rate = collected / elapsed * 60 if elapsed > 0 else 0

    # Final quota check
    final_remaining = check_remaining()

    log(f"\n{'='*60}")
    log(f"COLLECTION COMPLETE")
    log(f"{'='*60}")
    log(f"  Collected this session: {collected}")
    log(f"  Errors: {errors}")
    log(f"  Total in cache: {len(cache)}")
    log(f"  Rate: {rate:.0f} req/min")
    log(f"  Duration: {elapsed:.0f}s")
    log(f"  API requests remaining: {final_remaining}")

    # Coverage report
    for season in sorted(df["season"].unique()):
        s_df = df[df["season"] == season]
        s_cached = sum(1 for fid in s_df["fixture_id"].astype(int).astype(str) if fid in cache)
        log(f"  Season {season}: {s_cached}/{len(s_df)} ({s_cached/len(s_df)*100:.0f}%)")

    remaining_total = sum(
        1 for fid in df["fixture_id"].astype(int).astype(str) if fid not in cache
    )
    log(f"\n  Total remaining to collect: {remaining_total}")


if __name__ == "__main__":
    main()
