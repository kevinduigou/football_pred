import requests
import json
import time
from collections import Counter

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

# New seasons to collect (2017-2021)
NEW_SEASONS = [2017, 2018, 2019, 2020, 2021]

def fetch_fixtures(league_id, season):
    """Fetch all finished fixtures for a league/season."""
    params = {
        "league": league_id,
        "season": season,
        "status": "FT",
    }
    r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params=params)
    data = r.json()
    
    if "errors" in data and data["errors"]:
        print(f"  ERROR: {data['errors']}")
        return []
    
    fixtures = data.get("response", [])
    paging = data.get("paging", {})
    total_pages = paging.get("total", 1)
    
    # Handle pagination if needed
    if total_pages > 1:
        for page in range(2, total_pages + 1):
            params["page"] = page
            r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params=params)
            data = r.json()
            fixtures.extend(data.get("response", []))
            time.sleep(0.5)
    
    return fixtures

# Check current API usage
r = requests.get(f"{BASE_URL}/status", headers=HEADERS)
status = r.json()
current_requests = status["response"]["requests"]["current"]
limit = status["response"]["requests"]["limit_day"]
print(f"API usage: {current_requests}/{limit} requests today")
print(f"Need ~{len(LEAGUES) * len(NEW_SEASONS)} more requests\n")

# Collect all new data
new_data = []

for league_id, league_name in LEAGUES.items():
    for season in NEW_SEASONS:
        print(f"Fetching {league_name} {season}/{season+1}...")
        fixtures = fetch_fixtures(league_id, season)
        
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
            if match["home_goals"] is not None and match["away_goals"] is not None:
                if match["home_goals"] > match["away_goals"]:
                    match["result"] = "H"
                elif match["home_goals"] < match["away_goals"]:
                    match["result"] = "A"
                else:
                    match["result"] = "D"
                new_data.append(match)
        
        print(f"  Got {len(fixtures)} fixtures")
        time.sleep(1)

print(f"\n=== NEW DATA COLLECTED: {len(new_data)} matches (2017-2021) ===")

# Load existing data (2022-2024)
with open("/home/ubuntu/raw_fixtures.json", "r") as f:
    old_data = json.load(f)
print(f"Existing data: {len(old_data)} matches (2022-2024)")

# Merge
all_data = new_data + old_data

# Deduplicate by fixture_id
seen = set()
unique_data = []
for m in all_data:
    if m["fixture_id"] not in seen:
        seen.add(m["fixture_id"])
        unique_data.append(m)

print(f"Combined unique: {len(unique_data)} matches")

# Save combined raw data
with open("/home/ubuntu/raw_fixtures_extended.json", "w") as f:
    json.dump(unique_data, f, indent=2)
print("Saved to raw_fixtures_extended.json")

# Summary
league_counts = Counter(m["league_name"] for m in unique_data)
season_counts = Counter(m["season"] for m in unique_data)

print("\nBy league:")
for k, v in league_counts.most_common():
    print(f"  {k}: {v}")
print("\nBy season:")
for k, v in sorted(season_counts.items()):
    print(f"  {k}: {v}")
