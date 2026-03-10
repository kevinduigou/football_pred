import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from datetime import datetime

# Configuration
MODEL_PATH = "/home/ubuntu/football_pred/models/xgb_football_model_v4_advanced.pkl"
DATASET_PATH = "/home/ubuntu/football_pred/dataset/football_matches_v4.csv"
HOME_TEAM = "Rennes"
AWAY_TEAM = "Lille"
MATCH_DATE = "2026-03-08 17:00:00"

def get_team_stats(df, team_name, date):
    team_matches = df[((df['home_team'] == team_name) | (df['away_team'] == team_name)) & (df['date'] < date)].sort_values('date', ascending=False)
    if len(team_matches) == 0:
        return None
    return team_matches.iloc[0]

def prepare_features(df, home_team, away_team, match_date, model_features):
    home_last = get_team_stats(df, home_team, match_date)
    away_last = get_team_stats(df, away_team, match_date)
    
    if home_last is None or away_last is None:
        return None
    
    # Extract ELO and basic stats
    h_elo = home_last['home_elo'] if home_last['home_team'] == home_team else home_last['away_elo']
    a_elo = away_last['home_elo'] if away_last['home_team'] == away_team else away_last['away_elo']
    
    # Build feature dictionary
    feat_dict = {
        'home_elo': h_elo,
        'away_elo': a_elo,
        'elo_diff': h_elo - a_elo,
        'home_form_5': home_last['home_form_5'] if home_last['home_team'] == home_team else home_last['away_form_5'],
        'away_form_5': away_last['home_form_5'] if away_last['home_team'] == away_team else away_last['away_form_5'],
        'home_goals_for_avg': home_last['home_goals_for_avg'] if home_last['home_team'] == home_team else home_last['away_goals_for_avg'],
        'away_goals_for_avg': away_last['home_goals_for_avg'] if away_last['home_team'] == away_team else away_last['away_goals_for_avg'],
        'home_goals_against_avg': home_last['home_goals_against_avg'] if home_last['home_team'] == home_team else home_last['away_goals_against_avg'],
        'away_goals_against_avg': away_last['home_goals_against_avg'] if away_last['home_team'] == away_team else away_last['away_goals_against_avg'],
        'home_h2h_win_rate': home_last['home_h2h_win_rate'] if home_last['home_team'] == home_team else home_last['away_h2h_win_rate'],
        'away_h2h_win_rate': away_last['home_h2h_win_rate'] if away_last['home_team'] == away_team else away_last['away_h2h_win_rate'],
        'home_gd_form': home_last['home_gd_form'] if home_last['home_team'] == home_team else home_last['away_gd_form'],
        'away_gd_form': away_last['home_gd_form'] if away_last['home_team'] == away_team else away_last['away_gd_form'],
        'home_played_europe': home_last['home_played_europe'] if home_last['home_team'] == home_team else home_last['away_played_europe'],
        'away_played_europe': away_last['home_played_europe'] if away_last['home_team'] == away_team else away_last['away_played_europe'],
    }
    
    # Rest days
    match_dt = pd.to_datetime(match_date).tz_localize(None)
    h_last_dt = pd.to_datetime(home_last['date']).tz_localize(None)
    a_last_dt = pd.to_datetime(away_last['date']).tz_localize(None)
    feat_dict['home_rest_days'] = (match_dt - h_last_dt).days
    feat_dict['away_rest_days'] = (match_dt - a_last_dt).days
    
    # Advanced stats (rolling averages)
    adv_cols = [
        'shots_on_goal_avg5', 'total_shots_avg5', 'shots_insidebox_avg5',
        'ball_possession_avg5', 'total_passes_avg5', 'passes_pct_avg5',
        'corner_kicks_avg5', 'fouls_avg5', 'expected_goals_avg5'
    ]
    
    for col in adv_cols:
        feat_dict[f'home_{col}'] = home_last[f'home_{col}'] if home_last['home_team'] == home_team else home_last[f'away_{col}']
        feat_dict[f'away_{col}'] = away_last[f'home_{col}'] if away_last['home_team'] == away_team else away_last[f'away_{col}']
        
    # Reorder to match model
    X = pd.DataFrame([feat_dict])[model_features]
    return X, feat_dict

# Load model and data
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
model_features = model.get_booster().feature_names
df = pd.read_csv(DATASET_PATH)

# Predict
X, context = prepare_features(df, HOME_TEAM, AWAY_TEAM, MATCH_DATE, model_features)
probs = model.predict_proba(X)[0]

print(f"Prediction for {HOME_TEAM} vs {AWAY_TEAM} ({MATCH_DATE})")
print(f"Home Win: {probs[0]:.4f}")
print(f"Draw:     {probs[1]:.4f}")
print(f"Away Win: {probs[2]:.4f}")
print("\nContext:")
for k, v in context.items():
    print(f"  {k}: {v}")
