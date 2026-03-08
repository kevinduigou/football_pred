"""
Add European competition features to the football_matches dataset.

For each match in the dataset, check if the home team and/or the away team
played a European competition match (Champions League ID 2, Europa League ID 3,
Conference League ID 848) in the 7 days before the match.

Creates two new binary columns: home_played_europe, away_played_europe.
"""

import os
import time
import json
import requests
import pandas as pd
from datetime import timedelta
from collections import defaultdict

API_KEY = os.environ.get("FOOTBALL_API_SPORTS")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

EUROPEAN_LEAGUE_IDS = [2, 3, 848]  # UCL, UEL, UECL
SEASONS = list(range(2017, 2026))

CACHE_FILE = "/home/ubuntu/football_pred/dataset/european_fixtures_cache.json"

# --------------------------------------------------
# 1. Collect all European competition fixtures
# --------------------------------------------------
def fetch_european_fixtures():
    """Fetch all finished European competition fixtures for all seasons."""
    all_fixtures = []

    # Check if cache exists
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached European fixtures from {CACHE_FILE}")
        with open(CACHE_FILE, "r") as f:
            return json.load(f)

    # Check API status
    r = requests.get(f"{BASE_URL}/status", headers=HEADERS)
    status = r.json()
    current = status["response"]["requests"]["current"]
    limit = status["response"]["requests"]["limit_day"]
    print(f"API usage: {current}/{limit} requests today")

    total_requests = len(EUROPEAN_LEAGUE_IDS) * len(SEASONS)
    print(f"Need ~{total_requests} API requests for European fixtures")

    for league_id in EUROPEAN_LEAGUE_IDS:
        league_name = {2: "Champions League", 3: "Europa League", 848: "Conference League"}[league_id]
        for season in SEASONS:
            # Conference League only started in 2021
            if league_id == 848 and season < 2021:
                continue

            print(f"Fetching {league_name} {season}/{season+1}...")
            try:
                r = requests.get(
                    f"{BASE_URL}/fixtures",
                    headers=HEADERS,
                    params={
                        "league": league_id,
                        "season": season,
                        "status": "FT-AET-PEN",  # All finished statuses
                    },
                )
                data = r.json()

                if "errors" in data and data["errors"]:
                    print(f"  ERROR: {data['errors']}")
                    time.sleep(2)
                    continue

                fixtures = data.get("response", [])

                # Handle pagination
                paging = data.get("paging", {})
                total_pages = paging.get("total", 1)
                if total_pages > 1:
                    for page in range(2, total_pages + 1):
                        r = requests.get(
                            f"{BASE_URL}/fixtures",
                            headers=HEADERS,
                            params={
                                "league": league_id,
                                "season": season,
                                "status": "FT-AET-PEN",
                                "page": page,
                            },
                        )
                        page_data = r.json()
                        fixtures.extend(page_data.get("response", []))
                        time.sleep(0.5)

                for f in fixtures:
                    all_fixtures.append({
                        "fixture_id": f["fixture"]["id"],
                        "date": f["fixture"]["date"],
                        "league_id": league_id,
                        "league_name": league_name,
                        "season": season,
                        "home_team_id": f["teams"]["home"]["id"],
                        "home_team": f["teams"]["home"]["name"],
                        "away_team_id": f["teams"]["away"]["id"],
                        "away_team": f["teams"]["away"]["name"],
                    })

                print(f"  Got {len(fixtures)} fixtures")
                time.sleep(1)

            except Exception as e:
                print(f"  Exception: {e}")
                time.sleep(2)

    print(f"\n=== TOTAL EUROPEAN FIXTURES COLLECTED: {len(all_fixtures)} ===")

    # Save cache
    with open(CACHE_FILE, "w") as f:
        json.dump(all_fixtures, f, indent=2)
    print(f"Cached to {CACHE_FILE}")

    return all_fixtures


# --------------------------------------------------
# 2. Build lookup structure for European fixtures
# --------------------------------------------------
def build_europe_lookup(european_fixtures):
    """Build a dict: team_id -> list of match dates in European competitions."""
    team_europe_dates = defaultdict(list)

    for fix in european_fixtures:
        match_date = pd.to_datetime(fix["date"]).tz_localize(None)
        team_europe_dates[fix["home_team_id"]].append(match_date)
        team_europe_dates[fix["away_team_id"]].append(match_date)

    # Sort dates for each team
    for tid in team_europe_dates:
        team_europe_dates[tid] = sorted(team_europe_dates[tid])

    return team_europe_dates


def played_europe_in_last_7_days(team_id, match_date, team_europe_dates):
    """Check if a team played a European match in the 7 days before match_date."""
    if team_id not in team_europe_dates:
        return 0

    match_date_naive = pd.to_datetime(match_date)
    if match_date_naive.tzinfo is not None:
        match_date_naive = match_date_naive.tz_localize(None)

    window_start = match_date_naive - timedelta(days=7)

    for euro_date in team_europe_dates[team_id]:
        if window_start <= euro_date < match_date_naive:
            return 1

    return 0


# --------------------------------------------------
# 3. Main: enrich the dataset
# --------------------------------------------------
def main():
    print("=" * 60)
    print("ADDING EUROPEAN COMPETITION FEATURES")
    print("=" * 60)

    # Step 1: Collect European fixtures
    european_fixtures = fetch_european_fixtures()

    # Step 2: Build lookup
    team_europe_dates = build_europe_lookup(european_fixtures)

    # Show some stats
    teams_with_europe = len(team_europe_dates)
    total_europe_matches = sum(len(v) for v in team_europe_dates.values()) // 2
    print(f"\nTeams with European competition history: {teams_with_europe}")
    print(f"Total European matches: {total_europe_matches}")

    # Step 3: Load dataset
    dataset_path = "/home/ubuntu/football_pred/dataset/football_matches.csv"
    df = pd.read_csv(dataset_path)
    print(f"\nLoaded dataset: {df.shape[0]} matches, {df.shape[1]} columns")

    # We need team IDs to match. The dataset has team names but not IDs.
    # We need to build a name -> ID mapping from the European fixtures
    # AND from the original raw data.
    # Let's also use the team names in the dataset to find IDs.

    # Build name -> id mapping from European fixtures
    name_to_id = {}
    for fix in european_fixtures:
        name_to_id[fix["home_team"]] = fix["home_team_id"]
        name_to_id[fix["away_team"]] = fix["away_team_id"]

    print(f"Team name -> ID mapping from European data: {len(name_to_id)} teams")

    # Step 4: Add features
    home_played_europe = []
    away_played_europe = []

    unmatched_home = set()
    unmatched_away = set()

    for idx, row in df.iterrows():
        match_date = row["date"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        # Look up team IDs
        home_id = name_to_id.get(home_team)
        away_id = name_to_id.get(away_team)

        if home_id is not None:
            h_europe = played_europe_in_last_7_days(home_id, match_date, team_europe_dates)
        else:
            h_europe = 0  # Team not in European competitions
            if home_team not in unmatched_home:
                unmatched_home.add(home_team)

        if away_id is not None:
            a_europe = played_europe_in_last_7_days(away_id, match_date, team_europe_dates)
        else:
            a_europe = 0
            if away_team not in unmatched_away:
                unmatched_away.add(away_team)

        home_played_europe.append(h_europe)
        away_played_europe.append(a_europe)

    df["home_played_europe"] = home_played_europe
    df["away_played_europe"] = away_played_europe

    # Step 5: Save updated dataset
    df.to_csv(dataset_path, index=False)
    print(f"\nUpdated dataset saved: {df.shape[0]} matches, {df.shape[1]} columns")

    # Stats
    total_home_europe = sum(home_played_europe)
    total_away_europe = sum(away_played_europe)
    total_any_europe = sum(1 for h, a in zip(home_played_europe, away_played_europe) if h or a)

    print(f"\n=== EUROPEAN COMPETITION FEATURE STATS ===")
    print(f"Matches where home team played Europe in last 7 days: {total_home_europe} ({total_home_europe/len(df)*100:.1f}%)")
    print(f"Matches where away team played Europe in last 7 days: {total_away_europe} ({total_away_europe/len(df)*100:.1f}%)")
    print(f"Matches where at least one team played Europe:        {total_any_europe} ({total_any_europe/len(df)*100:.1f}%)")

    if unmatched_home:
        print(f"\nHome teams not found in European data (expected for non-European teams): {len(unmatched_home)}")
    if unmatched_away:
        print(f"Away teams not found in European data (expected for non-European teams): {len(unmatched_away)}")

    return df


if __name__ == "__main__":
    main()
