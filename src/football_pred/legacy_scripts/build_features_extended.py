import json
import pandas as pd
import numpy as np
from collections import defaultdict

# --------------------------------------------------
# 1. Load extended raw data
# --------------------------------------------------
with open("/home/ubuntu/raw_fixtures_extended.json", "r") as f:
    raw = json.load(f)

df = pd.DataFrame(raw)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"Total matches: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Seasons: {sorted(df['season'].unique())}")

# --------------------------------------------------
# 2. Compute ELO ratings
# --------------------------------------------------
K = 20
INITIAL_ELO = 1500
HOME_ADVANTAGE = 50

elo_ratings = defaultdict(lambda: INITIAL_ELO)

home_elos = []
away_elos = []
elo_diffs = []

for idx, row in df.iterrows():
    ht = row["home_team_id"]
    at = row["away_team_id"]
    
    home_elo = elo_ratings[ht]
    away_elo = elo_ratings[at]
    
    home_elos.append(home_elo)
    away_elos.append(away_elo)
    elo_diffs.append(home_elo - away_elo)
    
    exp_home = 1 / (1 + 10 ** ((away_elo - (home_elo + HOME_ADVANTAGE)) / 400))
    exp_away = 1 - exp_home
    
    if row["result"] == "H":
        actual_home, actual_away = 1.0, 0.0
    elif row["result"] == "A":
        actual_home, actual_away = 0.0, 1.0
    else:
        actual_home, actual_away = 0.5, 0.5
    
    elo_ratings[ht] += K * (actual_home - exp_home)
    elo_ratings[at] += K * (actual_away - exp_away)

df["home_elo"] = home_elos
df["away_elo"] = away_elos
df["elo_diff"] = elo_diffs
print("ELO ratings computed.")

# --------------------------------------------------
# 3. Compute rolling form (points in last 5 matches)
# --------------------------------------------------
FORM_WINDOW = 5
team_history = defaultdict(list)

home_form_5 = []
away_form_5 = []

for idx, row in df.iterrows():
    ht = row["home_team_id"]
    at = row["away_team_id"]
    
    h_hist = team_history[ht]
    a_hist = team_history[at]
    
    h_form = np.mean(h_hist[-FORM_WINDOW:]) if len(h_hist) >= FORM_WINDOW else (np.mean(h_hist) if h_hist else 1.0)
    a_form = np.mean(a_hist[-FORM_WINDOW:]) if len(a_hist) >= FORM_WINDOW else (np.mean(a_hist) if a_hist else 1.0)
    
    home_form_5.append(round(h_form, 3))
    away_form_5.append(round(a_form, 3))
    
    if row["result"] == "H":
        team_history[ht].append(3)
        team_history[at].append(0)
    elif row["result"] == "A":
        team_history[ht].append(0)
        team_history[at].append(3)
    else:
        team_history[ht].append(1)
        team_history[at].append(1)

df["home_form_5"] = home_form_5
df["away_form_5"] = away_form_5
print("Form features computed.")

# --------------------------------------------------
# 4. Compute rolling goals averages (last 5 matches)
# --------------------------------------------------
GOALS_WINDOW = 5
team_goals_for = defaultdict(list)
team_goals_against = defaultdict(list)

home_gf_avg = []
away_gf_avg = []
home_ga_avg = []
away_ga_avg = []

for idx, row in df.iterrows():
    ht = row["home_team_id"]
    at = row["away_team_id"]
    hg = row["home_goals"]
    ag = row["away_goals"]
    
    h_gf = team_goals_for[ht]
    h_ga = team_goals_against[ht]
    a_gf = team_goals_for[at]
    a_ga = team_goals_against[at]
    
    h_gf_avg = np.mean(h_gf[-GOALS_WINDOW:]) if len(h_gf) >= GOALS_WINDOW else (np.mean(h_gf) if h_gf else 1.3)
    a_gf_avg = np.mean(a_gf[-GOALS_WINDOW:]) if len(a_gf) >= GOALS_WINDOW else (np.mean(a_gf) if a_gf else 1.3)
    h_ga_avg = np.mean(h_ga[-GOALS_WINDOW:]) if len(h_ga) >= GOALS_WINDOW else (np.mean(h_ga) if h_ga else 1.1)
    a_ga_avg = np.mean(a_ga[-GOALS_WINDOW:]) if len(a_ga) >= GOALS_WINDOW else (np.mean(a_ga) if a_ga else 1.1)
    
    home_gf_avg.append(round(h_gf_avg, 3))
    away_gf_avg.append(round(a_gf_avg, 3))
    home_ga_avg.append(round(h_ga_avg, 3))
    away_ga_avg.append(round(a_ga_avg, 3))
    
    team_goals_for[ht].append(hg)
    team_goals_against[ht].append(ag)
    team_goals_for[at].append(ag)
    team_goals_against[at].append(hg)

df["home_goals_for_avg"] = home_gf_avg
df["away_goals_for_avg"] = away_gf_avg
df["home_goals_against_avg"] = home_ga_avg
df["away_goals_against_avg"] = away_ga_avg
print("Goals averages computed.")

# --------------------------------------------------
# 5. Compute rest days
# --------------------------------------------------
team_last_match = {}
home_rest = []
away_rest = []

for idx, row in df.iterrows():
    ht = row["home_team_id"]
    at = row["away_team_id"]
    match_date = row["date"]
    
    if ht in team_last_match:
        h_rest = (match_date - team_last_match[ht]).days
        h_rest = min(h_rest, 30)
    else:
        h_rest = 7
    
    if at in team_last_match:
        a_rest = (match_date - team_last_match[at]).days
        a_rest = min(a_rest, 30)
    else:
        a_rest = 7
    
    home_rest.append(h_rest)
    away_rest.append(a_rest)
    
    team_last_match[ht] = match_date
    team_last_match[at] = match_date

df["home_rest_days"] = home_rest
df["away_rest_days"] = away_rest
print("Rest days computed.")

# --------------------------------------------------
# 6. Head-to-head and goal difference form
# --------------------------------------------------
h2h_history = defaultdict(list)
home_h2h_wins = []
away_h2h_wins = []

for idx, row in df.iterrows():
    ht = row["home_team_id"]
    at = row["away_team_id"]
    key = (min(ht, at), max(ht, at))
    
    hist = h2h_history[key]
    if hist:
        recent = hist[-5:]
        h_wins = sum(1 for r in recent if r == ht) / len(recent)
        a_wins = sum(1 for r in recent if r == at) / len(recent)
    else:
        h_wins = 0.33
        a_wins = 0.33
    
    home_h2h_wins.append(round(h_wins, 3))
    away_h2h_wins.append(round(a_wins, 3))
    
    if row["result"] == "H":
        h2h_history[key].append(ht)
    elif row["result"] == "A":
        h2h_history[key].append(at)
    else:
        h2h_history[key].append(0)

df["home_h2h_win_rate"] = home_h2h_wins
df["away_h2h_win_rate"] = away_h2h_wins

# Goal difference form
team_gd_history = defaultdict(list)
home_gd_form = []
away_gd_form = []

for idx, row in df.iterrows():
    ht = row["home_team_id"]
    at = row["away_team_id"]
    
    h_gd = team_gd_history[ht]
    a_gd = team_gd_history[at]
    
    h_gd_avg = np.mean(h_gd[-FORM_WINDOW:]) if len(h_gd) >= FORM_WINDOW else (np.mean(h_gd) if h_gd else 0.0)
    a_gd_avg = np.mean(a_gd[-FORM_WINDOW:]) if len(a_gd) >= FORM_WINDOW else (np.mean(a_gd) if a_gd else 0.0)
    
    home_gd_form.append(round(h_gd_avg, 3))
    away_gd_form.append(round(a_gd_avg, 3))
    
    team_gd_history[ht].append(row["home_goals"] - row["away_goals"])
    team_gd_history[at].append(row["away_goals"] - row["home_goals"])

df["home_gd_form"] = home_gd_form
df["away_gd_form"] = away_gd_form
print("H2H and GD form computed.")

# --------------------------------------------------
# 7. Filter out warm-up period and save
# --------------------------------------------------
# With 8 seasons of data, use 500 matches as warm-up for stable ELO
WARMUP = 500
df_final = df.iloc[WARMUP:].copy()

output_cols = [
    "fixture_id", "date", "league_name", "season",
    "home_team", "away_team",
    "home_goals", "away_goals", "result",
    "home_elo", "away_elo", "elo_diff",
    "home_form_5", "away_form_5",
    "home_goals_for_avg", "away_goals_for_avg",
    "home_goals_against_avg", "away_goals_against_avg",
    "home_rest_days", "away_rest_days",
    "home_h2h_win_rate", "away_h2h_win_rate",
    "home_gd_form", "away_gd_form",
]

df_final = df_final[output_cols].reset_index(drop=True)

df_final.to_csv("/home/ubuntu/football_matches.csv", index=False)
print(f"\nSaved football_matches.csv with {len(df_final)} matches")
print(f"Dataset shape: {df_final.shape}")
print(f"Season range: {df_final['season'].min()} to {df_final['season'].max()}")
print(f"\nResult distribution:\n{df_final['result'].value_counts()}")
print(f"\nBy season:")
print(df_final.groupby("season").size())
print(f"\nFeature statistics:")
feature_cols = [c for c in output_cols if c not in ["fixture_id", "date", "league_name", "season", "home_team", "away_team", "home_goals", "away_goals", "result"]]
print(df_final[feature_cols].describe().round(3).to_string())

# Also save ELO ratings for all teams
team_names = {}
team_league = {}
team_match_count = defaultdict(int)
for _, row in df.iterrows():
    team_names[row["home_team_id"]] = row["home_team"]
    team_names[row["away_team_id"]] = row["away_team"]
    team_league[row["home_team_id"]] = row["league_name"]
    team_league[row["away_team_id"]] = row["league_name"]
    team_match_count[row["home_team_id"]] += 1
    team_match_count[row["away_team_id"]] += 1

elo_rows = []
for tid in team_names:
    elo_rows.append({
        "Team": team_names[tid],
        "League": team_league[tid],
        "Initial ELO": INITIAL_ELO,
        "Final ELO": round(elo_ratings[tid], 1),
        "ELO Change": round(elo_ratings[tid] - INITIAL_ELO, 1),
        "Matches": team_match_count[tid],
    })

elo_df = pd.DataFrame(elo_rows).sort_values("Final ELO", ascending=False)
elo_df.to_csv("/home/ubuntu/elo_ratings_all_teams.csv", index=False)
print(f"\nELO ratings saved for {len(elo_df)} teams")

# Show Rennes specifically
rennes = elo_df[elo_df["Team"].str.contains("Rennes", case=False)]
if not rennes.empty:
    print(f"\nRennes ELO: {rennes.iloc[0].to_dict()}")
