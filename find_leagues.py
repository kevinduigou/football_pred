import requests
import json

API_KEY = os.environ.get("FOOTBALL_API_SPORTS")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# Search for specific top leagues
target_leagues = [
    ("Premier League", "England"),
    ("La Liga", "Spain"),
    ("Serie A", "Italy"),
    ("Bundesliga", "Germany"),
    ("Ligue 1", "France"),
]

for name, country in target_leagues:
    r = requests.get(f"{BASE_URL}/leagues", headers=HEADERS, params={"name": name, "country": country})
    data = r.json()
    for lg in data["response"]:
        league = lg["league"]
        seasons = lg.get("seasons", [])
        season_years = [s["year"] for s in seasons]
        print(f"ID={league['id']:>4}  {country:>10} | {league['name']:<35} | seasons: {season_years[-5:] if len(season_years)>5 else season_years}")

# Also check fixtures endpoint structure
print("\n=== SAMPLE FIXTURES ===")
r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params={"league": 39, "season": 2023, "from": "2024-01-01", "to": "2024-01-31"})
data = r.json()
print(f"Results: {data['results']}")
if data["response"]:
    print(json.dumps(data["response"][0], indent=2))
