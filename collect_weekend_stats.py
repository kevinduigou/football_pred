"""
Collecte des statistiques avancées pour les 34 nouveaux matchs du week-end 7-9 mars 2026.
"""
import requests, json, time, os

API_KEY = os.environ.get('FOOTBALL_API_SPORTS', '')
CACHE_PATH = '/home/ubuntu/football_pred/dataset/advanced_stats_cache.json'
HEADERS = {'x-apisports-key': API_KEY}
BASE_URL = 'https://v3.football.api-sports.io'

STAT_KEYS = [
    'Shots on Goal', 'Total Shots', 'Shots insidebox', 'Shots outsidebox',
    'Ball Possession', 'Total passes', 'Passes accurate', 'Passes %',
    'Corner Kicks', 'Fouls', 'expected_goals'
]

FIXTURE_IDS = [
    1378134, 1378135, 1378136, 1378137, 1378138, 1378139, 1378140, 1378141, 1378142,
    1387914, 1387915, 1387916, 1387917, 1387918, 1387919, 1387920, 1387922,
    1388525, 1388526, 1388527, 1388528, 1388529, 1388530, 1388531, 1388532,
    1391079, 1391080, 1391082, 1391083, 1391084, 1391085, 1391086, 1391087, 1391088
]

# Load existing cache
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, 'r') as f:
        cache = json.load(f)
else:
    cache = {}

# Filter out already cached
to_collect = [fid for fid in FIXTURE_IDS if str(fid) not in cache]
print(f"Cache existant: {len(cache)} matchs")
print(f"Nouveaux matchs à collecter: {len(to_collect)} sur {len(FIXTURE_IDS)}")

collected = 0
errors = 0

for i, fid in enumerate(to_collect):
    try:
        url = f"{BASE_URL}/fixtures/statistics"
        resp = requests.get(url, headers=HEADERS, params={'fixture': fid}, timeout=10)
        data = resp.json()

        if 'response' in data and len(data['response']) >= 2:
            home_stats = {}
            away_stats = {}
            for stat in data['response'][0].get('statistics', []):
                if stat['type'] in STAT_KEYS:
                    home_stats[stat['type']] = stat['value']
            for stat in data['response'][1].get('statistics', []):
                if stat['type'] in STAT_KEYS:
                    away_stats[stat['type']] = stat['value']

            cache[str(fid)] = {
                'home_team': data['response'][0].get('team', {}).get('name', ''),
                'away_team': data['response'][1].get('team', {}).get('name', ''),
                'home_stats': home_stats,
                'away_stats': away_stats
            }
            collected += 1
        else:
            errors += 1
            print(f"  WARN: fixture {fid} - pas de données stats")

    except Exception as e:
        errors += 1
        print(f"  ERR: fixture {fid} - {e}")

    if (i + 1) % 10 == 0:
        print(f"  Progression: {i+1}/{len(to_collect)} (collectés={collected}, erreurs={errors})")

    time.sleep(0.25)  # Rate limiting

# Save cache
with open(CACHE_PATH, 'w') as f:
    json.dump(cache, f)

print(f"\nTerminé: {collected} collectés, {errors} erreurs")
print(f"Cache total: {len(cache)} matchs")
print(f"Cache sauvegardé: {CACHE_PATH}")
