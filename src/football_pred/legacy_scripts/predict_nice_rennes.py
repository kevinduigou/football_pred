import json
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Load ALL raw data (2017-2024 + 2025)
# --------------------------------------------------
with open("/home/ubuntu/raw_fixtures_extended.json", "r") as f:
    data_2017_2024 = json.load(f)

with open("/home/ubuntu/raw_fixtures_2025.json", "r") as f:
    data_2025 = json.load(f)

# Merge and deduplicate
all_raw = data_2017_2024 + data_2025
seen = set()
unique_data = []
for m in all_raw:
    if m["fixture_id"] not in seen:
        seen.add(m["fixture_id"])
        unique_data.append(m)

df = pd.DataFrame(unique_data)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"Total matches (all seasons): {len(df)}")
print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
print(f"Seasons: {sorted(df['season'].unique())}")
print(f"By season:\n{df.groupby('season').size()}")

# --------------------------------------------------
# 2. Compute ALL features from scratch (ELO, form, etc.)
# --------------------------------------------------
K = 20
INITIAL_ELO = 1500
HOME_ADVANTAGE = 50
FORM_WINDOW = 5
GOALS_WINDOW = 5

# Initialize trackers
elo_ratings = defaultdict(lambda: INITIAL_ELO)
team_history = defaultdict(list)
team_goals_for = defaultdict(list)
team_goals_against = defaultdict(list)
team_last_match = {}
h2h_history = defaultdict(list)
team_gd_history = defaultdict(list)

# Storage for features
features = {
    "home_elo": [], "away_elo": [], "elo_diff": [],
    "home_form_5": [], "away_form_5": [],
    "home_goals_for_avg": [], "away_goals_for_avg": [],
    "home_goals_against_avg": [], "away_goals_against_avg": [],
    "home_rest_days": [], "away_rest_days": [],
    "home_h2h_win_rate": [], "away_h2h_win_rate": [],
    "home_gd_form": [], "away_gd_form": [],
}

for idx, row in df.iterrows():
    ht = row["home_team_id"]
    at = row["away_team_id"]
    hg = row["home_goals"]
    ag = row["away_goals"]
    match_date = row["date"]
    
    # --- ELO ---
    home_elo = elo_ratings[ht]
    away_elo = elo_ratings[at]
    features["home_elo"].append(home_elo)
    features["away_elo"].append(away_elo)
    features["elo_diff"].append(home_elo - away_elo)
    
    # --- Form ---
    h_hist = team_history[ht]
    a_hist = team_history[at]
    h_form = np.mean(h_hist[-FORM_WINDOW:]) if len(h_hist) >= FORM_WINDOW else (np.mean(h_hist) if h_hist else 1.0)
    a_form = np.mean(a_hist[-FORM_WINDOW:]) if len(a_hist) >= FORM_WINDOW else (np.mean(a_hist) if a_hist else 1.0)
    features["home_form_5"].append(round(h_form, 3))
    features["away_form_5"].append(round(a_form, 3))
    
    # --- Goals avg ---
    h_gf = team_goals_for[ht]
    h_ga = team_goals_against[ht]
    a_gf = team_goals_for[at]
    a_ga = team_goals_against[at]
    features["home_goals_for_avg"].append(round(np.mean(h_gf[-GOALS_WINDOW:]) if len(h_gf) >= GOALS_WINDOW else (np.mean(h_gf) if h_gf else 1.3), 3))
    features["away_goals_for_avg"].append(round(np.mean(a_gf[-GOALS_WINDOW:]) if len(a_gf) >= GOALS_WINDOW else (np.mean(a_gf) if a_gf else 1.3), 3))
    features["home_goals_against_avg"].append(round(np.mean(h_ga[-GOALS_WINDOW:]) if len(h_ga) >= GOALS_WINDOW else (np.mean(h_ga) if h_ga else 1.1), 3))
    features["away_goals_against_avg"].append(round(np.mean(a_ga[-GOALS_WINDOW:]) if len(a_ga) >= GOALS_WINDOW else (np.mean(a_ga) if a_ga else 1.1), 3))
    
    # --- Rest days ---
    h_rest = min((match_date - team_last_match[ht]).days, 30) if ht in team_last_match else 7
    a_rest = min((match_date - team_last_match[at]).days, 30) if at in team_last_match else 7
    features["home_rest_days"].append(h_rest)
    features["away_rest_days"].append(a_rest)
    
    # --- H2H ---
    key = (min(ht, at), max(ht, at))
    hist = h2h_history[key]
    if hist:
        recent = hist[-5:]
        h_wins = sum(1 for r in recent if r == ht) / len(recent)
        a_wins = sum(1 for r in recent if r == at) / len(recent)
    else:
        h_wins, a_wins = 0.33, 0.33
    features["home_h2h_win_rate"].append(round(h_wins, 3))
    features["away_h2h_win_rate"].append(round(a_wins, 3))
    
    # --- GD form ---
    h_gd = team_gd_history[ht]
    a_gd = team_gd_history[at]
    features["home_gd_form"].append(round(np.mean(h_gd[-FORM_WINDOW:]) if len(h_gd) >= FORM_WINDOW else (np.mean(h_gd) if h_gd else 0.0), 3))
    features["away_gd_form"].append(round(np.mean(a_gd[-FORM_WINDOW:]) if len(a_gd) >= FORM_WINDOW else (np.mean(a_gd) if a_gd else 0.0), 3))
    
    # === UPDATE ALL TRACKERS ===
    # ELO update
    exp_home = 1 / (1 + 10 ** ((away_elo - (home_elo + HOME_ADVANTAGE)) / 400))
    exp_away = 1 - exp_home
    if row["result"] == "H":
        actual_home, actual_away = 1.0, 0.0
        team_history[ht].append(3); team_history[at].append(0)
        h2h_history[key].append(ht)
    elif row["result"] == "A":
        actual_home, actual_away = 0.0, 1.0
        team_history[ht].append(0); team_history[at].append(3)
        h2h_history[key].append(at)
    else:
        actual_home, actual_away = 0.5, 0.5
        team_history[ht].append(1); team_history[at].append(1)
        h2h_history[key].append(0)
    
    elo_ratings[ht] += K * (actual_home - exp_home)
    elo_ratings[at] += K * (actual_away - exp_away)
    
    team_goals_for[ht].append(hg); team_goals_against[ht].append(ag)
    team_goals_for[at].append(ag); team_goals_against[at].append(hg)
    team_last_match[ht] = match_date; team_last_match[at] = match_date
    team_gd_history[ht].append(hg - ag); team_gd_history[at].append(ag - hg)

# Add features to dataframe
for col, vals in features.items():
    df[col] = vals

print("\nAll features computed for all matches including 2025-2026.")

# --------------------------------------------------
# 3. Save updated football_matches.csv
# --------------------------------------------------
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
print(f"Updated football_matches.csv: {len(df_final)} matches")
print(f"Seasons: {sorted(df_final['season'].unique())}")
print(f"By season:\n{df_final.groupby('season').size()}")

# --------------------------------------------------
# 4. Current state of Nice and Rennes
# --------------------------------------------------
NICE_ID = 84
RENNES_ID = 94

print("\n" + "="*60)
print("CURRENT STATE: NICE (ID=84)")
print("="*60)
print(f"  ELO: {elo_ratings[NICE_ID]:.1f}")
print(f"  Form (last 5): {np.mean(team_history[NICE_ID][-5:]):.2f} pts/game")
print(f"  Goals for avg (last 5): {np.mean(team_goals_for[NICE_ID][-5:]):.2f}")
print(f"  Goals against avg (last 5): {np.mean(team_goals_against[NICE_ID][-5:]):.2f}")
print(f"  GD form (last 5): {np.mean(team_gd_history[NICE_ID][-5:]):.2f}")
print(f"  Last match: {team_last_match[NICE_ID].strftime('%Y-%m-%d')}")
print(f"  Last 5 results (pts): {team_history[NICE_ID][-5:]}")
print(f"  Last 5 goals scored: {team_goals_for[NICE_ID][-5:]}")
print(f"  Last 5 goals conceded: {team_goals_against[NICE_ID][-5:]}")

print(f"\n{'='*60}")
print("CURRENT STATE: RENNES (ID=94)")
print("="*60)
print(f"  ELO: {elo_ratings[RENNES_ID]:.1f}")
print(f"  Form (last 5): {np.mean(team_history[RENNES_ID][-5:]):.2f} pts/game")
print(f"  Goals for avg (last 5): {np.mean(team_goals_for[RENNES_ID][-5:]):.2f}")
print(f"  Goals against avg (last 5): {np.mean(team_goals_against[RENNES_ID][-5:]):.2f}")
print(f"  GD form (last 5): {np.mean(team_gd_history[RENNES_ID][-5:]):.2f}")
print(f"  Last match: {team_last_match[RENNES_ID].strftime('%Y-%m-%d')}")
print(f"  Last 5 results (pts): {team_history[RENNES_ID][-5:]}")
print(f"  Last 5 goals scored: {team_goals_for[RENNES_ID][-5:]}")
print(f"  Last 5 goals conceded: {team_goals_against[RENNES_ID][-5:]}")

# H2H
h2h_key = (min(NICE_ID, RENNES_ID), max(NICE_ID, RENNES_ID))
h2h = h2h_history[h2h_key]
print(f"\n{'='*60}")
print("HEAD-TO-HEAD HISTORY")
print("="*60)
h2h_recent = h2h[-5:]
nice_h2h_wins = sum(1 for r in h2h_recent if r == NICE_ID) / len(h2h_recent) if h2h_recent else 0.33
rennes_h2h_wins = sum(1 for r in h2h_recent if r == RENNES_ID) / len(h2h_recent) if h2h_recent else 0.33
print(f"  Last {len(h2h_recent)} meetings: Nice wins={sum(1 for r in h2h_recent if r==NICE_ID)}, Rennes wins={sum(1 for r in h2h_recent if r==RENNES_ID)}, Draws={sum(1 for r in h2h_recent if r==0)}")
print(f"  Nice H2H win rate: {nice_h2h_wins:.3f}")
print(f"  Rennes H2H win rate: {rennes_h2h_wins:.3f}")

# Rest days (match is on 2026-03-08)
from datetime import datetime
match_date = pd.Timestamp("2026-03-08", tz="UTC")
nice_rest = (match_date - team_last_match[NICE_ID]).days
rennes_rest = (match_date - team_last_match[RENNES_ID]).days
print(f"\n  Nice rest days: {nice_rest} (last match: {team_last_match[NICE_ID].strftime('%Y-%m-%d')})")
print(f"  Rennes rest days: {rennes_rest} (last match: {team_last_match[RENNES_ID].strftime('%Y-%m-%d')})")

# --------------------------------------------------
# 5. Build feature vector for Nice vs Rennes
# --------------------------------------------------
match_features = {
    "home_elo": elo_ratings[NICE_ID],
    "away_elo": elo_ratings[RENNES_ID],
    "elo_diff": elo_ratings[NICE_ID] - elo_ratings[RENNES_ID],
    "home_form_5": round(np.mean(team_history[NICE_ID][-5:]), 3),
    "away_form_5": round(np.mean(team_history[RENNES_ID][-5:]), 3),
    "home_goals_for_avg": round(np.mean(team_goals_for[NICE_ID][-5:]), 3),
    "away_goals_for_avg": round(np.mean(team_goals_for[RENNES_ID][-5:]), 3),
    "home_goals_against_avg": round(np.mean(team_goals_against[NICE_ID][-5:]), 3),
    "away_goals_against_avg": round(np.mean(team_goals_against[RENNES_ID][-5:]), 3),
    "home_rest_days": min(nice_rest, 30),
    "away_rest_days": min(rennes_rest, 30),
    "home_h2h_win_rate": round(nice_h2h_wins, 3),
    "away_h2h_win_rate": round(rennes_h2h_wins, 3),
    "home_gd_form": round(np.mean(team_gd_history[NICE_ID][-5:]), 3),
    "away_gd_form": round(np.mean(team_gd_history[RENNES_ID][-5:]), 3),
}

print(f"\n{'='*60}")
print("FEATURE VECTOR: Nice (Home) vs Rennes (Away)")
print("="*60)
for k, v in match_features.items():
    print(f"  {k:>25}: {v}")

# --------------------------------------------------
# 6. Load model and predict
# --------------------------------------------------
with open("/home/ubuntu/xgb_football_model.pkl", "rb") as f:
    model = pickle.load(f)

feature_cols = [
    "home_elo", "away_elo", "elo_diff",
    "home_form_5", "away_form_5",
    "home_goals_for_avg", "away_goals_for_avg",
    "home_goals_against_avg", "away_goals_against_avg",
    "home_rest_days", "away_rest_days",
    "home_h2h_win_rate", "away_h2h_win_rate",
    "home_gd_form", "away_gd_form",
]

match_df = pd.DataFrame([match_features])[feature_cols]
proba = model.predict_proba(match_df)[0]

print(f"\n{'='*60}")
print("PREDICTION: Nice vs Rennes — 8 March 2026")
print("Venue: Allianz Riviera, Nice")
print("="*60)
print(f"\n  P(Nice Win):  {proba[0]:.4f}  ({proba[0]*100:.1f}%)")
print(f"  P(Draw):      {proba[1]:.4f}  ({proba[1]*100:.1f}%)")
print(f"  P(Rennes Win):{proba[2]:.4f}  ({proba[2]*100:.1f}%)")
print(f"\n  Implied Odds:")
print(f"    Nice:   {1/proba[0]:.2f}")
print(f"    Draw:   {1/proba[1]:.2f}")
print(f"    Rennes: {1/proba[2]:.2f}")

# Most likely outcome
outcomes = ["Nice Win", "Draw", "Rennes Win"]
most_likely = outcomes[np.argmax(proba)]
print(f"\n  Most likely outcome: {most_likely} ({max(proba)*100:.1f}%)")

# --------------------------------------------------
# 7. Show Nice and Rennes recent form detail
# --------------------------------------------------
print(f"\n{'='*60}")
print("NICE — Last 5 matches (2025-2026)")
print("="*60)
nice_recent = df[(df["home_team_id"] == NICE_ID) | (df["away_team_id"] == NICE_ID)].tail(5)
for _, m in nice_recent.iterrows():
    is_home = m["home_team_id"] == NICE_ID
    opponent = m["away_team"] if is_home else m["home_team"]
    venue = "H" if is_home else "A"
    score = f"{m['home_goals']}-{m['away_goals']}"
    print(f"  {m['date'].strftime('%Y-%m-%d')} ({venue}) vs {opponent}: {score} [{m['result']}]")

print(f"\n{'='*60}")
print("RENNES — Last 5 matches (2025-2026)")
print("="*60)
rennes_recent = df[(df["home_team_id"] == RENNES_ID) | (df["away_team_id"] == RENNES_ID)].tail(5)
for _, m in rennes_recent.iterrows():
    is_home = m["home_team_id"] == RENNES_ID
    opponent = m["away_team"] if is_home else m["home_team"]
    venue = "H" if is_home else "A"
    score = f"{m['home_goals']}-{m['away_goals']}"
    print(f"  {m['date'].strftime('%Y-%m-%d')} ({venue}) vs {opponent}: {score} [{m['result']}]")
