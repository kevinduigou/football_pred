import requests
import json
import time

API_KEY = "c1b837c37df33c46d475f5a67c346c22"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# All 5 leagues, season 2025
LEAGUES = {
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
}

# Check API status
r = requests.get(f"{BASE_URL}/status", headers=HEADERS)
status = r.json()
print(f"API usage: {status['response']['requests']['current']}/{status['response']['requests']['limit_day']}")

# Collect finished matches for 2025 season
new_data = []
for league_id, league_name in LEAGUES.items():
    print(f"\nFetching {league_name} 2025/2026...")
    r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params={
        "league": league_id,
        "season": 2025,
        "status": "FT",
    })
    data = r.json()
    fixtures = data.get("response", [])
    
    for f in fixtures:
        match = {
            "fixture_id": f["fixture"]["id"],
            "date": f["fixture"]["date"],
            "league_id": league_id,
            "league_name": league_name,
            "season": 2025,
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
    
    print(f"  Got {len(fixtures)} finished fixtures")
    time.sleep(1)

print(f"\n=== 2025-2026 SEASON: {len(new_data)} finished matches ===")

# Save
with open("/home/ubuntu/raw_fixtures_2025.json", "w") as f:
    json.dump(new_data, f, indent=2)

# Find Nice and Rennes
nice_matches = [m for m in new_data if "Nice" in m["home_team"] or "Nice" in m["away_team"]]
rennes_matches = [m for m in new_data if "Rennes" in m["home_team"] or "Rennes" in m["away_team"]]

print(f"\nNice matches this season: {len(nice_matches)}")
if nice_matches:
    for m in nice_matches[-5:]:
        print(f"  {m['date'][:10]}: {m['home_team']} {m['home_goals']}-{m['away_goals']} {m['away_team']} ({m['result']})")
    nice_id = nice_matches[0]["home_team_id"] if "Nice" in nice_matches[0]["home_team"] else nice_matches[0]["away_team_id"]
    print(f"  Nice team ID: {nice_id}")

print(f"\nRennes matches this season: {len(rennes_matches)}")
if rennes_matches:
    for m in rennes_matches[-5:]:
        print(f"  {m['date'][:10]}: {m['home_team']} {m['home_goals']}-{m['away_goals']} {m['away_team']} ({m['result']})")
    rennes_id = rennes_matches[0]["home_team_id"] if "Rennes" in rennes_matches[0]["home_team"] else rennes_matches[0]["away_team_id"]
    print(f"  Rennes team ID: {rennes_id}")

# Also check for upcoming Nice vs Rennes fixture
print("\n=== Checking upcoming Nice vs Rennes ===")
r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params={
    "league": 61,
    "season": 2025,
    "status": "NS",  # Not started
})
data = r.json()
upcoming = data.get("response", [])
print(f"Total upcoming Ligue 1 fixtures: {len(upcoming)}")

for f in upcoming:
    home = f["teams"]["home"]["name"]
    away = f["teams"]["away"]["name"]
    if ("Nice" in home and "Rennes" in away) or ("Rennes" in home and "Nice" in away):
        print(f"  FOUND: {f['fixture']['date'][:10]} - {home} vs {away} (Fixture ID: {f['fixture']['id']})")
        print(f"  Venue: {f['fixture']['venue']['name']}")

# Also check next few fixtures for both teams
print("\n=== Next fixtures for Nice ===")
r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params={
    "league": 61,
    "season": 2025,
    "status": "NS",
    "team": nice_id if nice_matches else 84,
    "next": 5,
})
data = r.json()
for f in data.get("response", []):
    print(f"  {f['fixture']['date'][:10]}: {f['teams']['home']['name']} vs {f['teams']['away']['name']}")

print("\n=== Next fixtures for Rennes ===")
r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params={
    "league": 61,
    "season": 2025,
    "status": "NS",
    "team": rennes_id if rennes_matches else 94,
    "next": 5,
})
data = r.json()
for f in data.get("response", []):
    print(f"  {f['fixture']['date'][:10]}: {f['teams']['home']['name']} vs {f['teams']['away']['name']}")
