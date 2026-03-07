import requests
import json

API_KEY = "c1b837c37df33c46d475f5a67c346c22"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# 1. Check account status / quota
r = requests.get(f"{BASE_URL}/status", headers=HEADERS)
print("=== ACCOUNT STATUS ===")
print(json.dumps(r.json(), indent=2))

# 2. List some popular leagues
r = requests.get(f"{BASE_URL}/leagues", headers=HEADERS, params={"type": "league", "current": "true"})
data = r.json()
print(f"\n=== LEAGUES (total: {data['results']}) ===")
for lg in data["response"][:30]:
    league = lg["league"]
    country = lg["country"]
    seasons = lg.get("seasons", [])
    latest = seasons[-1]["year"] if seasons else "?"
    print(f"  ID={league['id']:>4}  {country['name']:>20} | {league['name']:<40} | latest season: {latest}")
