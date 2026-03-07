import requests
import json

API_KEY = "c1b837c37df33c46d475f5a67c346c22"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# 1. Check available bookmakers
print("=== AVAILABLE BOOKMAKERS ===")
r = requests.get(f"{BASE_URL}/odds/bookmakers", headers=HEADERS)
data = r.json()
bookmakers = data.get("response", [])
print(f"Total bookmakers: {len(bookmakers)}")

betclic_found = False
for bm in bookmakers:
    name = bm.get("name", "")
    if "betclic" in name.lower() or "bet clic" in name.lower():
        betclic_found = True
        print(f"  *** BETCLIC FOUND: ID={bm['id']}, Name={bm['name']} ***")

# Show all bookmakers to find Betclic or similar
print("\nAll bookmakers:")
for bm in bookmakers:
    marker = " <-- BETCLIC" if "betclic" in bm["name"].lower() else ""
    print(f"  ID={bm['id']:3d} | {bm['name']}{marker}")

if not betclic_found:
    print("\n*** Betclic NOT found in bookmakers list ***")

# 2. Check available bet types
print("\n=== AVAILABLE BET TYPES ===")
r = requests.get(f"{BASE_URL}/odds/bets", headers=HEADERS)
data = r.json()
bets = data.get("response", [])
print(f"Total bet types: {len(bets)}")
for b in bets[:20]:
    print(f"  ID={b['id']:3d} | {b['name']}")

# 3. Try to get odds for Nice vs Rennes (fixture 1387920)
FIXTURE_ID = 1387920
print(f"\n=== ODDS FOR FIXTURE {FIXTURE_ID} (Nice vs Rennes) ===")
r = requests.get(f"{BASE_URL}/odds", headers=HEADERS, params={
    "fixture": FIXTURE_ID,
})
data = r.json()
odds_data = data.get("response", [])
print(f"Odds responses: {len(odds_data)}")

if odds_data:
    for entry in odds_data:
        league = entry.get("league", {})
        print(f"\nLeague: {league.get('name')} (Season {league.get('season')})")
        for bm in entry.get("bookmakers", []):
            bm_name = bm.get("name", "")
            is_betclic = "betclic" in bm_name.lower()
            marker = " *** BETCLIC ***" if is_betclic else ""
            print(f"\n  Bookmaker: {bm_name}{marker}")
            for bet in bm.get("bets", []):
                print(f"    Bet: {bet['name']}")
                for val in bet.get("values", []):
                    print(f"      {val['value']}: {val['odd']}")
else:
    print("No odds available yet for this fixture.")
    # Try with league/season instead
    print("\nTrying with league/season params...")
    r = requests.get(f"{BASE_URL}/odds", headers=HEADERS, params={
        "league": 61,
        "season": 2025,
        "page": 1,
    })
    data = r.json()
    odds_data = data.get("response", [])
    print(f"Odds responses for Ligue 1 2025: {len(odds_data)}")
    if odds_data:
        # Check if any entry has Betclic
        for entry in odds_data[:3]:
            fixture = entry.get("fixture", {})
            print(f"\n  Fixture: {fixture}")
            for bm in entry.get("bookmakers", [])[:3]:
                print(f"    Bookmaker: {bm['name']}")

# 4. Also check live odds endpoint
print(f"\n=== LIVE ODDS ===")
r = requests.get(f"{BASE_URL}/odds/live", headers=HEADERS, params={
    "fixture": FIXTURE_ID,
})
data = r.json()
print(f"Live odds response: {json.dumps(data.get('response', [])[:1], indent=2)[:500]}")

# 5. Check pre-match odds mapping
print(f"\n=== ODDS MAPPING ===")
r = requests.get(f"{BASE_URL}/odds/mapping", headers=HEADERS, params={
    "page": 1,
})
data = r.json()
mapping = data.get("response", [])
print(f"Mapping entries: {len(mapping)}")
if mapping:
    # Look for our fixture
    for m in mapping:
        if m.get("fixture", {}).get("id") == FIXTURE_ID:
            print(f"  Found fixture {FIXTURE_ID}: {m}")
            break
    # Show a few entries
    for m in mapping[:5]:
        print(f"  {m}")

print(f"\n=== API STATUS ===")
r = requests.get(f"{BASE_URL}/status", headers=HEADERS)
status = r.json()
print(f"Requests used: {status['response']['requests']['current']}/{status['response']['requests']['limit_day']}")
