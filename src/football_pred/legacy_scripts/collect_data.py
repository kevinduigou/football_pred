import requests
import json
import time
import os
import csv

API_KEY = os.environ.get("FOOTBALL_API_SPORTS")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# Top 5 European leagues
LEAGUES = {
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
}

# Seasons to collect (3 full seasons + current)
SEASONS = [2022, 2023, 2024]

OUTPUT_FILE = "/home/ubuntu/raw_fixtures.json"

def fetch_fixtures(league_id, season):
    """Fetch all finished fixtures for a league/season."""
    all_fixtures = []
    page = 1
    while True:
        params = {
            "league": league_id,
            "season": season,
            "status": "FT",  # only finished matches
        }
        r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params=params)
        data = r.json()
        
        if "errors" in data and data["errors"]:
            print(f"  ERROR: {data['errors']}")
            break
            
        fixtures = data.get("response", [])
        all_fixtures.extend(fixtures)
        
        paging = data.get("paging", {})
        current_page = paging.get("current", 1)
        total_pages = paging.get("total", 1)
        
        print(f"    Page {current_page}/{total_pages}, got {len(fixtures)} fixtures")
        
        if current_page >= total_pages:
            break
        page += 1
        time.sleep(0.5)  # Rate limiting
    
    return all_fixtures

# Collect all data
all_data = []
request_count = 0

for league_id, league_name in LEAGUES.items():
    for season in SEASONS:
        print(f"\nFetching {league_name} {season}/{season+1}...")
        fixtures = fetch_fixtures(league_id, season)
        request_count += 1
        
        for f in fixtures:
            match = {
                "fixture_id": f["fixture"]["id"],
                "date": f["fixture"]["date"],
                "league_id": league_id,
                "league_name": league_name,
                "season": season,
                "home_team_id": f["teams"]["home"]["id"],
                "home_team": f["teams"]["home"]["name"],
                "away_team_id": f["teams"]["away"]["id"],
                "away_team": f["teams"]["away"]["name"],
                "home_goals": f["goals"]["home"],
                "away_goals": f["goals"]["away"],
                "home_ht": f["score"]["halftime"]["home"],
                "away_ht": f["score"]["halftime"]["away"],
            }
            # Determine result
            if match["home_goals"] > match["away_goals"]:
                match["result"] = "H"
            elif match["home_goals"] < match["away_goals"]:
                match["result"] = "A"
            else:
                match["result"] = "D"
            
            all_data.append(match)
        
        print(f"  Total fixtures: {len(fixtures)}")
        time.sleep(1)  # Rate limiting between requests

print(f"\n=== TOTAL MATCHES COLLECTED: {len(all_data)} ===")
print(f"API requests used: ~{request_count}")

# Save raw data
with open(OUTPUT_FILE, "w") as f:
    json.dump(all_data, f, indent=2)
print(f"Saved to {OUTPUT_FILE}")

# Quick summary
from collections import Counter
league_counts = Counter(m["league_name"] for m in all_data)
season_counts = Counter(m["season"] for m in all_data)
result_counts = Counter(m["result"] for m in all_data)

print("\nBy league:")
for k, v in league_counts.most_common():
    print(f"  {k}: {v}")
print("\nBy season:")
for k, v in sorted(season_counts.items()):
    print(f"  {k}: {v}")
print("\nBy result:")
for k, v in result_counts.most_common():
    print(f"  {k}: {v}")
