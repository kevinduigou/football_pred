"""
Fast concurrent collection of advanced match statistics from API-Football.
Uses ThreadPoolExecutor for concurrent requests to maximize throughput.
Pro plan: 450 req/min limit. We use 4 concurrent workers with 0.15s delay.
"""
import json
import os
import sys
import time
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

API_KEY = os.environ.get("FOOTBALL_API_SPORTS", "")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
CACHE_FILE = "/home/ubuntu/football_pred/dataset/advanced_stats_cache.json"
DATASET_FILE = "/home/ubuntu/football_pred/dataset/football_matches.csv"
LOG_FILE = "/tmp/collect_fast.log"

PRIORITY_SEASONS = [2024, 2023, 2022, 2021, 2020, 2019]
STAT_KEYS = [
    "Shots on Goal", "Total Shots", "Shots insidebox", "Shots outsidebox",
    "Ball Possession", "Total passes", "Passes accurate", "Passes %",
    "Corner Kicks", "Fouls", "expected_goals",
]
QUOTA_BUFFER = 20
WORKERS = 3  # concurrent workers
SAVE_EVERY = 25  # save cache every N matches

cache_lock = threading.Lock()
log_lock = threading.Lock()

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with log_lock:
        print(line, flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")

def check_remaining():
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
    r = requests.get(
        f"{BASE_URL}/fixtures/statistics",
        headers=HEADERS,
        params={"fixture": fixture_id},
        timeout=15,
    )
    data = r.json()
    if data.get("results", 0) == 0:
        return fixture_id, {}
    result = {}
    for team_data in data["response"]:
        team_id = str(team_data["team"]["id"])
        team_name = team_data["team"]["name"]
        stats = {"team_name": team_name}
        for s in team_data.get("statistics", []):
            if s["type"] in STAT_KEYS:
                stats[s["type"]] = s["value"]
        for s in team_data.get("statistics", []):
            if "expected" in s["type"].lower() and "goal" in s["type"].lower():
                stats["expected_goals"] = s["value"]
        result[team_id] = stats
    return fixture_id, result if result else {}

def main():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
    else:
        cache = {}
    log(f"Cache loaded: {len(cache)} entries")

    df = pd.read_csv(DATASET_FILE)
    df["date"] = pd.to_datetime(df["date"])
    log(f"Dataset: {len(df)} matches")

    remaining = check_remaining()
    log(f"API requests remaining: {remaining}")
    if remaining is not None and remaining <= QUOTA_BUFFER:
        log("Quota exhausted. Stopping.")
        return

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
        log(f"  Season {season}: {len(season_ids)} to collect ({cached_count} cached)")

    log(f"Total to collect: {len(to_collect)}")
    if not to_collect:
        log("Nothing to collect!")
        return

    collected = 0
    errors = 0
    start = time.time()
    requests_used = 0
    stop_flag = False

    batch_size = 50
    for batch_start in range(0, len(to_collect), batch_size):
        if stop_flag:
            break

        # Check quota every batch
        if requests_used > 0 and requests_used % 200 == 0:
            rem = check_remaining()
            requests_used += 1
            if rem is not None and rem <= QUOTA_BUFFER:
                log(f"Quota nearly exhausted ({rem} remaining). Stopping.")
                stop_flag = True
                break

        batch = to_collect[batch_start:batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {}
            for fid, season in batch:
                if stop_flag:
                    break
                time.sleep(0.14)  # ~7 req/s across workers = ~420/min
                futures[executor.submit(fetch_stats, fid)] = (fid, season)

            for future in as_completed(futures):
                try:
                    fid, stats = future.result(timeout=30)
                    requests_used += 1
                    with cache_lock:
                        cache[fid] = stats
                    collected += 1
                except Exception as e:
                    errors += 1
                    fid_info = futures[future]
                    log(f"  Error on fixture {fid_info[0]}: {e}")

        # Save cache after each batch
        with cache_lock:
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)

        elapsed = time.time() - start
        rate = collected / elapsed * 60 if elapsed > 0 else 0
        log(f"Progress: {collected}/{len(to_collect)} collected, {errors} errors, "
            f"{rate:.0f} req/min, cache={len(cache)} entries")

    # Final save
    with cache_lock:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)

    elapsed = time.time() - start
    rate = collected / elapsed * 60 if elapsed > 0 else 0
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
