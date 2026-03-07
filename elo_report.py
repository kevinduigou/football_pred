import json
import pandas as pd
import numpy as np
from collections import defaultdict

# --------------------------------------------------
# Rebuild ELO exactly as in build_features.py
# --------------------------------------------------
with open("/home/ubuntu/raw_fixtures.json", "r") as f:
    raw = json.load(f)

df = pd.DataFrame(raw)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

K = 20
INITIAL_ELO = 1500
HOME_ADVANTAGE = 50

elo_ratings = defaultdict(lambda: INITIAL_ELO)

# Track first appearance and team names
team_names = {}
team_first_match = {}
team_match_count = defaultdict(int)
team_league = {}

for idx, row in df.iterrows():
    ht = row["home_team_id"]
    at = row["away_team_id"]
    
    # Record team info
    if ht not in team_names:
        team_names[ht] = row["home_team"]
        team_first_match[ht] = row["date"]
        team_league[ht] = row["league_name"]
    if at not in team_names:
        team_names[at] = row["away_team"]
        team_first_match[at] = row["date"]
        team_league[at] = row["league_name"]
    
    team_match_count[ht] += 1
    team_match_count[at] += 1
    
    home_elo = elo_ratings[ht]
    away_elo = elo_ratings[at]
    
    # Expected scores
    exp_home = 1 / (1 + 10 ** ((away_elo - (home_elo + HOME_ADVANTAGE)) / 400))
    exp_away = 1 - exp_home
    
    # Actual scores
    if row["result"] == "H":
        actual_home, actual_away = 1.0, 0.0
    elif row["result"] == "A":
        actual_home, actual_away = 0.0, 1.0
    else:
        actual_home, actual_away = 0.5, 0.5
    
    # Update ELOs
    elo_ratings[ht] += K * (actual_home - exp_home)
    elo_ratings[at] += K * (actual_away - exp_away)

# --------------------------------------------------
# Build summary table
# --------------------------------------------------
rows = []
for tid in team_names:
    rows.append({
        "Team": team_names[tid],
        "League": team_league[tid],
        "Team ID": tid,
        "Initial ELO": INITIAL_ELO,
        "Final ELO": round(elo_ratings[tid], 1),
        "ELO Change": round(elo_ratings[tid] - INITIAL_ELO, 1),
        "Matches": team_match_count[tid],
    })

elo_df = pd.DataFrame(rows).sort_values("Final ELO", ascending=False).reset_index(drop=True)

# --------------------------------------------------
# Print by league
# --------------------------------------------------
print("=" * 90)
print(f"ALL TEAMS START WITH INITIAL ELO = {INITIAL_ELO}")
print("=" * 90)
print(f"\nThe ELO system uses K={K} and a home advantage of {HOME_ADVANTAGE} points.")
print(f"Every team begins at {INITIAL_ELO} and their rating evolves match by match.\n")

for league in ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]:
    league_df = elo_df[elo_df["League"] == league].copy()
    league_df = league_df.sort_values("Final ELO", ascending=False).reset_index(drop=True)
    league_df.index += 1
    print(f"\n{'='*70}")
    print(f"  {league}")
    print(f"{'='*70}")
    print(league_df[["Team", "Initial ELO", "Final ELO", "ELO Change", "Matches"]].to_string())
    print()

# Top 20 overall
print(f"\n{'='*70}")
print(f"  TOP 20 TEAMS BY FINAL ELO (ALL LEAGUES)")
print(f"{'='*70}")
top20 = elo_df.head(20).copy()
top20.index = range(1, 21)
print(top20[["Team", "League", "Initial ELO", "Final ELO", "ELO Change", "Matches"]].to_string())

# Bottom 20
print(f"\n{'='*70}")
print(f"  BOTTOM 20 TEAMS BY FINAL ELO (ALL LEAGUES)")
print(f"{'='*70}")
bot20 = elo_df.tail(20).copy()
bot20.index = range(1, 21)
print(bot20[["Team", "League", "Initial ELO", "Final ELO", "ELO Change", "Matches"]].to_string())

# Save full table
elo_df.to_csv("/home/ubuntu/elo_ratings_all_teams.csv", index=False)
print(f"\nFull table saved to elo_ratings_all_teams.csv ({len(elo_df)} teams)")
