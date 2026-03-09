"""
Build rolling average features (last 5 matches) from advanced match statistics.

For each team, computes rolling averages of key stats over their last 5 matches
(regardless of home/away), then assigns them as home_X_avg5 / away_X_avg5.

IMPORTANT: Only uses data from matches BEFORE the current match (no data leakage).
"""

import pandas as pd
import numpy as np
import json

DATASET_PATH = "/home/ubuntu/football_pred/dataset/football_matches.csv"
STATS_PATH = "/home/ubuntu/football_pred/dataset/advanced_stats.csv"
OUTPUT_PATH = "/home/ubuntu/football_pred/dataset/football_matches_v4.csv"

ROLLING_WINDOW = 5

# Stats to compute rolling averages for
ROLLING_STATS = [
    "shots_on_goal",
    "total_shots",
    "shots_insidebox",
    "ball_possession",
    "total_passes",
    "passes_pct",       # passes accuracy %
    "corner_kicks",
    "fouls",
    "expected_goals",
]


def main():
    print("=" * 60)
    print("BUILDING ROLLING AVERAGE FEATURES")
    print("=" * 60)

    # Load datasets
    df = pd.read_csv(DATASET_PATH)
    stats = pd.read_csv(STATS_PATH)
    print(f"Main dataset: {df.shape}")
    print(f"Stats dataset: {stats.shape}")

    # Merge stats into main dataset
    df = df.merge(stats, on="fixture_id", how="left")
    print(f"After merge: {df.shape}")

    # Ensure date is datetime and sort chronologically
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Build a unified match history per team
    # For each match, a team has stats regardless of home/away
    # We need to create a "team-centric" view

    # Step 1: Create team-match records (one per team per match)
    records = []
    for idx, row in df.iterrows():
        fid = row["fixture_id"]
        date = row["date"]
        league = row["league_name"]
        season = row["season"]

        # Home team record
        home_rec = {
            "fixture_id": fid,
            "date": date,
            "team": row["home_team"],
            "league": league,
            "season": season,
        }
        for stat in ROLLING_STATS:
            home_rec[stat] = row.get(f"home_{stat}")
        records.append(home_rec)

        # Away team record
        away_rec = {
            "fixture_id": fid,
            "date": date,
            "team": row["away_team"],
            "league": league,
            "season": season,
        }
        for stat in ROLLING_STATS:
            away_rec[stat] = row.get(f"away_{stat}")
        records.append(away_rec)

    team_df = pd.DataFrame(records)
    team_df = team_df.sort_values("date").reset_index(drop=True)
    print(f"Team-match records: {team_df.shape}")

    # Clean percentage columns: remove '%' suffix and convert to numeric
    for stat in ROLLING_STATS:
        if team_df[stat].dtype == object:
            team_df[stat] = team_df[stat].astype(str).str.rstrip('%').replace('nan', None)
            team_df[stat] = pd.to_numeric(team_df[stat], errors='coerce')

    # Step 2: For each team, compute rolling averages (shift to avoid leakage)
    print("Computing rolling averages per team...")
    rolling_cols = {}
    for stat in ROLLING_STATS:
        col_name = f"{stat}_avg{ROLLING_WINDOW}"
        # Group by team, sort by date, compute rolling mean with min_periods=1
        # shift(1) ensures we only use PAST matches
        team_df[col_name] = (
            team_df.groupby("team")[stat]
            .transform(lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=1).mean())
        )
        rolling_cols[stat] = col_name

    print(f"Rolling columns created: {list(rolling_cols.values())}")

    # Step 3: Split back into home and away, then merge into main df
    # Create lookup: (fixture_id, team) -> rolling stats
    home_lookup = {}
    away_lookup = {}

    for idx, row in df.iterrows():
        fid = row["fixture_id"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        # Find the home team's rolling stats for this fixture
        mask_home = (team_df["fixture_id"] == fid) & (team_df["team"] == home_team)
        mask_away = (team_df["fixture_id"] == fid) & (team_df["team"] == away_team)

        home_rows = team_df[mask_home]
        away_rows = team_df[mask_away]

        for stat, col in rolling_cols.items():
            col_home = f"home_{stat}_avg{ROLLING_WINDOW}"
            col_away = f"away_{stat}_avg{ROLLING_WINDOW}"

            if len(home_rows) > 0:
                df.loc[idx, col_home] = home_rows.iloc[0][col]
            else:
                df.loc[idx, col_home] = np.nan

            if len(away_rows) > 0:
                df.loc[idx, col_away] = away_rows.iloc[0][col]
            else:
                df.loc[idx, col_away] = np.nan

    # Show coverage
    new_cols = [f"home_{s}_avg{ROLLING_WINDOW}" for s in ROLLING_STATS] + \
               [f"away_{s}_avg{ROLLING_WINDOW}" for s in ROLLING_STATS]

    print(f"\n--- Coverage Summary ---")
    for col in new_cols:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null}/{len(df)} ({non_null/len(df)*100:.1f}%)")

    # Drop the raw per-match stats columns (keep only rolling averages)
    raw_cols = [c for c in df.columns if c.startswith("home_") or c.startswith("away_")]
    raw_stat_cols = [c for c in raw_cols if any(c.endswith(s) for s in [
        "shots_on_goal", "shots_off_goal", "total_shots", "blocked_shots",
        "shots_insidebox", "shots_outsidebox", "fouls", "corner_kicks",
        "offsides", "ball_possession", "yellow_cards", "red_cards",
        "goalkeeper_saves", "total_passes", "passes_accurate", "passes_pct",
        "expected_goals", "goals_prevented"
    ]) and "_avg" not in c]

    df = df.drop(columns=raw_stat_cols, errors="ignore")

    print(f"\nFinal dataset: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
