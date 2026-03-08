"""
Build advanced_stats.csv from the cache file.
Maps API team IDs to dataset team names to ensure correct home/away assignment.
"""

import json
import pandas as pd

DATASET_PATH = "/home/ubuntu/football_pred/dataset/football_matches.csv"
CACHE_FILE = "/home/ubuntu/football_pred/dataset/advanced_stats_cache.json"
OUTPUT_FILE = "/home/ubuntu/football_pred/dataset/advanced_stats.csv"

STAT_TYPES = [
    "Shots on Goal", "Shots off Goal", "Total Shots", "Blocked Shots",
    "Shots insidebox", "Shots outsidebox",
    "Fouls", "Corner Kicks", "Offsides",
    "Ball Possession", "Yellow Cards", "Red Cards",
    "Goalkeeper Saves", "Total passes", "Passes accurate", "Passes %",
    "expected_goals", "goals_prevented",
]


def clean_col_name(stat_name):
    return stat_name.lower().replace(" ", "_").replace("%", "pct")


def match_team(cache_entry, dataset_home, dataset_away):
    """Match API team names to dataset team names.
    Returns (home_stats, away_stats) dicts."""
    team_ids = list(cache_entry.keys())
    if len(team_ids) < 2:
        return {}, {}

    t1_stats = cache_entry[team_ids[0]]
    t2_stats = cache_entry[team_ids[1]]
    t1_name = t1_stats.get("team_name", "")
    t2_name = t2_stats.get("team_name", "")

    # Try exact match first
    if t1_name == dataset_home or t2_name == dataset_away:
        return t1_stats, t2_stats
    if t2_name == dataset_home or t1_name == dataset_away:
        return t2_stats, t1_stats

    # Fuzzy: check if one name contains the other
    dh = dataset_home.lower()
    da = dataset_away.lower()
    t1l = t1_name.lower()
    t2l = t2_name.lower()

    if t1l in dh or dh in t1l:
        return t1_stats, t2_stats
    if t2l in dh or dh in t2l:
        return t2_stats, t1_stats

    # Default: API returns home first
    return t1_stats, t2_stats


def main():
    with open(CACHE_FILE) as f:
        cache = json.load(f)
    print(f"Cache entries: {len(cache)}")

    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset: {df.shape}")

    rows = []
    matched = 0
    for _, row in df.iterrows():
        fid = str(int(row["fixture_id"]))
        entry = cache.get(fid, {})
        rd = {"fixture_id": int(fid)}

        if entry and len(entry) >= 2:
            hs, aws = match_team(entry, row["home_team"], row["away_team"])
            for sn in STAT_TYPES:
                cn = clean_col_name(sn)
                rd[f"home_{cn}"] = hs.get(sn)
                rd[f"away_{cn}"] = aws.get(sn)
            matched += 1
        else:
            for sn in STAT_TYPES:
                cn = clean_col_name(sn)
                rd[f"home_{cn}"] = None
                rd[f"away_{cn}"] = None

        rows.append(rd)

    sdf = pd.DataFrame(rows)
    sdf.to_csv(OUTPUT_FILE, index=False)

    print(f"Matched: {matched}, Unmatched: {len(df) - matched}")
    print(f"Saved to: {OUTPUT_FILE}")

    # Quick summary of coverage
    wd = sdf.dropna(subset=["home_shots_on_goal"]).shape[0]
    xg = sdf.dropna(subset=["home_expected_goals"]).shape[0]
    print(f"With shot stats: {wd}")
    print(f"With xG: {xg}")


if __name__ == "__main__":
    main()
