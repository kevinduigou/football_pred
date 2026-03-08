"""
Collect advanced match statistics from API-Football /fixtures/statistics endpoint.
Pro plan: 7500 requests/day, ~300 requests/minute.
Uses cache file for resuming. Saves cache every 100 requests.
"""

import requests
import json
import time
import os
import pandas as pd

API_KEY = os.environ.get("FOOTBALL_API_SPORTS")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

DATASET_PATH = "/home/ubuntu/football_pred/dataset/football_matches.csv"
CACHE_FILE = "/home/ubuntu/football_pred/dataset/advanced_stats_cache.json"
OUTPUT_FILE = "/home/ubuntu/football_pred/dataset/advanced_stats.csv"


def check_api_status():
    r = requests.get(f"{BASE_URL}/status", headers=HEADERS, timeout=10)
    data = r.json()
    resp = data.get("response", {})
    req = resp.get("requests", {})
    cur = req.get("current", 0)
    lim = req.get("limit_day", 7500)
    print(f"Plan: {resp.get('subscription', {}).get('plan', '?')}")
    print(f"Requests: {cur}/{lim}")
    return cur, lim


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def parse_stat_value(value):
    if value is None:
        return None
    if isinstance(value, str) and "%" in value:
        return float(value.replace("%", ""))
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def fetch_fixture_stats(fixture_id):
    try:
        r = requests.get(
            f"{BASE_URL}/fixtures/statistics",
            headers=HEADERS,
            params={"fixture": fixture_id},
            timeout=15,
        )
        data = r.json()
        if data.get("errors"):
            return None, str(data["errors"])
        response = data.get("response", [])
        if not response:
            return {}, "empty"
        result = {}
        for team_data in response:
            tid = str(team_data.get("team", {}).get("id"))
            tname = team_data.get("team", {}).get("name")
            stats = {"team_name": tname}
            for s in team_data.get("statistics", []):
                stats[s["type"]] = parse_stat_value(s.get("value"))
            result[tid] = stats
        return result, None
    except Exception as e:
        return None, str(e)


def main():
    print("=" * 60)
    print("COLLECTING ADVANCED MATCH STATISTICS")
    print("=" * 60)

    cur, lim = check_api_status()
    remaining = lim - cur
    print(f"Remaining: {remaining}")

    df = pd.read_csv(DATASET_PATH)
    fixture_ids = df["fixture_id"].unique().tolist()
    print(f"Total fixtures: {len(fixture_ids)}")

    cache = load_cache()
    to_fetch = [fid for fid in fixture_ids if str(fid) not in cache]
    print(f"Cached: {len(cache)}, To fetch: {len(to_fetch)}")

    if to_fetch:
        batch = min(len(to_fetch), remaining - 10)
        if batch <= 0:
            print("No API calls remaining.")
        else:
            print(f"Fetching {batch} fixtures...")
            ok = err = empty = 0
            t0 = time.time()

            for i, fid in enumerate(to_fetch[:batch]):
                stats, error = fetch_fixture_stats(int(fid))
                fid_s = str(fid)

                if stats is not None and len(stats) > 0:
                    cache[fid_s] = stats
                    ok += 1
                elif stats is not None:
                    cache[fid_s] = {}
                    empty += 1
                else:
                    err += 1
                    if error and ("rate" in error.lower() or "429" in error):
                        print(f"  Rate limited at {i+1}, waiting 60s...")
                        time.sleep(60)
                        stats, error = fetch_fixture_stats(int(fid))
                        if stats is not None and len(stats) > 0:
                            cache[fid_s] = stats
                            ok += 1; err -= 1
                        elif stats is not None:
                            cache[fid_s] = {}
                            empty += 1; err -= 1

                if (i + 1) % 100 == 0:
                    save_cache(cache)
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed * 60
                    eta = (batch - i - 1) / rate if rate > 0 else 0
                    print(f"  [{i+1}/{batch}] OK={ok} Empty={empty} Err={err} | {rate:.0f}/min | ETA: {eta:.0f}min")

                # ~250 req/min to be safe with Pro plan
                time.sleep(0.24)

            save_cache(cache)
            elapsed = time.time() - t0
            print(f"\nDone in {elapsed/60:.1f}min. OK={ok} Empty={empty} Err={err}")

    # Build CSV
    print("\nBuilding output CSV...")
    stat_types = [
        "Shots on Goal", "Shots off Goal", "Total Shots", "Blocked Shots",
        "Shots insidebox", "Shots outsidebox",
        "Fouls", "Corner Kicks", "Offsides",
        "Ball Possession", "Yellow Cards", "Red Cards",
        "Goalkeeper Saves", "Total passes", "Passes accurate", "Passes %",
        "expected_goals", "goals_prevented",
    ]

    rows = []
    for _, row in df.iterrows():
        fid = str(int(row["fixture_id"]))
        stats = cache.get(fid, {})
        rd = {"fixture_id": int(fid)}

        if stats and len(stats) >= 2:
            tids = list(stats.keys())
            hs = stats.get(tids[0], {})
            aws = stats.get(tids[1], {})
            for sn in stat_types:
                cn = sn.lower().replace(" ", "_").replace("%", "pct")
                rd[f"home_{cn}"] = hs.get(sn)
                rd[f"away_{cn}"] = aws.get(sn)
        else:
            for sn in stat_types:
                cn = sn.lower().replace(" ", "_").replace("%", "pct")
                rd[f"home_{cn}"] = None
                rd[f"away_{cn}"] = None
        rows.append(rd)

    sdf = pd.DataFrame(rows)
    sdf.to_csv(OUTPUT_FILE, index=False)

    total = len(sdf)
    wd = sdf.dropna(subset=["home_shots_on_goal"]).shape[0]
    print(f"Total: {total}, With stats: {wd} ({wd/total*100:.1f}%)")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
