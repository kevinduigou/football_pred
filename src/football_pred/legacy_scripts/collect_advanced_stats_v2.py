"""
Optimized advanced stats collection script (v2).

Strategy:
1. Use /fixtures?league=X&season=Y&status=FT to get all fixture IDs in bulk
   (1 call per league/season instead of scanning CSV)
2. Skip fixtures already in cache
3. Prioritize recent seasons (2025, 2024, 2023, ...) for test set coverage
4. Maximize throughput at 300 req/min (Pro plan limit)
5. Save cache every 100 fixtures for safety
"""

import requests
import json
import time
import os
import sys
import pandas as pd

API_KEY = os.environ.get("FOOTBALL_API_SPORTS", "")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

CACHE_FILE = "/home/ubuntu/football_pred/dataset/advanced_stats_cache.json"
DATASET_FILE = "/home/ubuntu/football_pred/dataset/football_matches.csv"

# Leagues and their API IDs
LEAGUES = {
    "Premier League": 39,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61,
}

# Seasons to collect, ordered by priority (recent first for test set)
SEASONS_PRIORITY = [2025, 2024, 2023, 2022, 2021, 2020, 2019]

# Rate limiting: Pro plan = 300 req/min
REQUEST_DELAY = 0.21  # ~285 req/min, safe margin

STAT_TYPES = [
    "Shots on Goal", "Shots off Goal", "Total Shots", "Blocked Shots",
    "Shots insidebox", "Shots outsidebox",
    "Fouls", "Corner Kicks", "Offsides",
    "Ball Possession", "Yellow Cards", "Red Cards",
    "Goalkeeper Saves", "Total passes", "Passes accurate", "Passes %",
    "expected_goals", "goals_prevented",
]


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def get_api_status():
    r = requests.get(f"{BASE_URL}/status", headers=HEADERS, timeout=10)
    data = r.json()
    req = data.get("response", {}).get("requests", {})
    return req.get("current", 0), req.get("limit_day", 7500)


def get_fixture_ids_for_league_season(league_id, season):
    """Get all finished fixture IDs for a league/season in ONE API call."""
    r = requests.get(
        f"{BASE_URL}/fixtures",
        headers=HEADERS,
        params={"league": league_id, "season": season, "status": "FT"},
        timeout=30,
    )
    data = r.json()
    if data.get("errors"):
        print(f"  Error: {data['errors']}")
        return []
    fixtures = []
    for fix in data.get("response", []):
        fid = fix["fixture"]["id"]
        home_id = fix["teams"]["home"]["id"]
        away_id = fix["teams"]["away"]["id"]
        home_name = fix["teams"]["home"]["name"]
        away_name = fix["teams"]["away"]["name"]
        fixtures.append({
            "fixture_id": fid,
            "home_id": home_id,
            "away_id": away_id,
            "home_name": home_name,
            "away_name": away_name,
        })
    return fixtures


def get_fixture_stats(fixture_id):
    """Get statistics for a single fixture."""
    r = requests.get(
        f"{BASE_URL}/fixtures/statistics",
        headers=HEADERS,
        params={"fixture": fixture_id},
        timeout=15,
    )
    data = r.json()
    if data.get("errors") or not data.get("response"):
        return None

    result = {}
    for team_data in data["response"]:
        team_id = str(team_data["team"]["id"])
        team_name = team_data["team"]["name"]
        stats = {"team_name": team_name}
        for stat in team_data.get("statistics", []):
            stype = stat["type"]
            val = stat["value"]
            if isinstance(val, str) and "%" in val:
                val = float(val.replace("%", ""))
            elif val is not None:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    pass
            stats[stype] = val
        result[team_id] = stats
    return result


def main():
    cache = load_cache()
    print(f"Cache loaded: {len(cache)} fixtures")

    # Load dataset to know which fixture IDs we need
    df = pd.read_csv(DATASET_FILE)
    dataset_fids = set(df["fixture_id"].astype(int).astype(str).tolist())
    print(f"Dataset fixtures: {len(dataset_fids)}")

    current, limit = get_api_status()
    remaining_quota = limit - current
    print(f"API quota: {current}/{limit} (remaining: {remaining_quota})")

    # Phase 1: Get all fixture IDs per league/season (bulk)
    print("\n=== Phase 1: Bulk fixture ID retrieval ===")
    all_to_collect = []
    for season in SEASONS_PRIORITY:
        for league_name, league_id in LEAGUES.items():
            # Check if we have any uncached fixtures for this combo
            season_mask = df[(df["season"] == season) & (df["league_name"] == league_name)]
            uncached = [
                str(int(fid)) for fid in season_mask["fixture_id"]
                if str(int(fid)) not in cache
            ]
            if not uncached:
                continue

            print(f"  {league_name} {season}: {len(uncached)} uncached fixtures, fetching IDs...")
            fixtures = get_fixture_ids_for_league_season(league_id, season)
            time.sleep(REQUEST_DELAY)

            # Filter to only fixtures in our dataset AND not in cache
            for fix in fixtures:
                fid_str = str(fix["fixture_id"])
                if fid_str in dataset_fids and fid_str not in cache:
                    all_to_collect.append(fix)

    print(f"\nTotal fixtures to collect: {len(all_to_collect)}")
    print(f"Remaining API quota: {remaining_quota - len(LEAGUES) * len(SEASONS_PRIORITY)}")

    if not all_to_collect:
        print("Nothing to collect!")
        return

    # Phase 2: Collect statistics for each fixture
    print("\n=== Phase 2: Collecting fixture statistics ===")
    ok_count = 0
    empty_count = 0
    err_count = 0
    start_time = time.time()

    for i, fix in enumerate(all_to_collect):
        fid = fix["fixture_id"]
        fid_str = str(fid)

        # Skip if already cached (race condition safety)
        if fid_str in cache:
            continue

        # Check quota
        if ok_count + empty_count + err_count > 0 and (ok_count + empty_count + err_count) % 500 == 0:
            curr, lim = get_api_status()
            if curr >= lim - 10:
                print(f"\n  QUOTA LIMIT REACHED ({curr}/{lim}). Stopping.")
                break

        try:
            stats = get_fixture_stats(fid)
            if stats and len(stats) >= 2:
                cache[fid_str] = stats
                ok_count += 1
            elif stats:
                empty_count += 1
            else:
                empty_count += 1
        except Exception as e:
            err_count += 1
            if err_count <= 5:
                print(f"  Error for {fid}: {e}")

        time.sleep(REQUEST_DELAY)

        # Progress logging
        total_done = ok_count + empty_count + err_count
        if total_done % 100 == 0 and total_done > 0:
            elapsed = time.time() - start_time
            rate = total_done / (elapsed / 60)
            remaining = len(all_to_collect) - i - 1
            eta = remaining / rate if rate > 0 else 0
            save_cache(cache)
            print(f"  [{total_done}/{len(all_to_collect)}] OK={ok_count} Empty={empty_count} Err={err_count} | {rate:.0f}/min | ETA: {eta:.0f}min | Cache: {len(cache)}")

    # Final save
    save_cache(cache)
    elapsed = time.time() - start_time
    print(f"\n=== Collection Complete ===")
    print(f"Collected: OK={ok_count}, Empty={empty_count}, Errors={err_count}")
    print(f"Total cache: {len(cache)} fixtures")
    print(f"Time: {elapsed/60:.1f} minutes")

    # Coverage report
    cached_set = set(cache.keys())
    print(f"\n=== Coverage Report ===")
    for season in sorted(df["season"].unique()):
        mask = df["season"] == season
        total_s = mask.sum()
        has_s = sum(1 for fid in df[mask]["fixture_id"] if str(int(fid)) in cached_set)
        print(f"  {season}: {has_s}/{total_s} ({has_s/total_s*100:.0f}%)")


if __name__ == "__main__":
    main()
