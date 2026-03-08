"""
Prédiction : Rennes (domicile) vs Nice (extérieur)
Modèle : XGBoost v4 (avec statistiques avancées)
Date du match : 2026-03-08
"""

import json
import pickle
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paramètres ────────────────────────────────────────────────────────────
HOME_TEAM = "Nice"
AWAY_TEAM = "Rennes"
MATCH_DATE = pd.Timestamp("2026-03-08").tz_localize(None)
FORM_WINDOW = 5
K = 20
INITIAL_ELO = 1500
HOME_ADVANTAGE = 50

DATASET   = "/home/ubuntu/football_pred/dataset/football_matches.csv"
CACHE     = "/home/ubuntu/football_pred/dataset/advanced_stats_cache.json"
MODEL_PKL = "/home/ubuntu/football_pred/models/xgb_football_model_v4_advanced.pkl"

ROLL_STATS = [
    "shots_on_goal", "total_shots", "shots_insidebox",
    "ball_possession", "total_passes", "passes_pct",
    "corner_kicks", "fouls", "expected_goals",
]

RAW_STAT_MAP = {
    "shots_on_goal":  "Shots on Goal",
    "total_shots":    "Total Shots",
    "shots_insidebox":"Shots insidebox",
    "ball_possession":"Ball Possession",
    "total_passes":   "Total passes",
    "passes_pct":     "Passes %",
    "corner_kicks":   "Corner Kicks",
    "fouls":          "Fouls",
    "expected_goals": "expected_goals",
}

# ── 1. Charger le dataset ─────────────────────────────────────────────────
df = pd.read_csv(DATASET)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"Dataset chargé : {len(df)} matchs ({df['date'].min().date()} → {df['date'].max().date()})")

# ── 2. Charger le cache de stats avancées ─────────────────────────────────
with open(CACHE) as f:
    cache = json.load(f)
print(f"Cache stats avancées : {len(cache)} matchs")

# ── 3. Recalculer ELO et features de base à partir du dataset ─────────────
elo = defaultdict(lambda: INITIAL_ELO)
history = defaultdict(list)        # points par match
goals_for = defaultdict(list)
goals_against = defaultdict(list)
last_match = {}
h2h = defaultdict(list)
gd_hist = defaultdict(list)
# Stats avancées par équipe (liste de valeurs brutes)
adv_stats = defaultdict(lambda: defaultdict(list))

for _, row in df.iterrows():
    ht = row["home_team"]
    at = row["away_team"]
    hg = row["home_goals"]
    ag = row["away_goals"]
    fid = str(int(row["fixture_id"]))

    # Récupérer stats avancées du cache si disponibles
    entry = cache.get(fid, {})
    if entry and len(entry) >= 2:
        tids = list(entry.keys())
        t1, t2 = entry[tids[0]], entry[tids[1]]
        n1 = t1.get("team_name", "")
        n2 = t2.get("team_name", "")
        if n1.lower() in ht.lower() or ht.lower() in n1.lower():
            hs, aws = t1, t2
        elif n2.lower() in ht.lower() or ht.lower() in n2.lower():
            hs, aws = t2, t1
        else:
            hs, aws = t1, t2  # défaut : API retourne domicile en premier

        for stat_key, raw_key in RAW_STAT_MAP.items():
            hv = hs.get(raw_key)
            av = aws.get(raw_key)
            if hv is not None:
                try:
                    adv_stats[ht][stat_key].append(float(str(hv).replace("%", "").strip()))
                except (ValueError, TypeError):
                    pass
            if av is not None:
                try:
                    adv_stats[at][stat_key].append(float(str(av).replace("%", "").strip()))
                except (ValueError, TypeError):
                    pass

    # Mettre à jour les trackers APRÈS avoir lu les features
    h_elo = elo[ht]
    a_elo = elo[at]
    exp_h = 1 / (1 + 10 ** ((a_elo - (h_elo + HOME_ADVANTAGE)) / 400))
    exp_a = 1 - exp_h
    key = tuple(sorted([ht, at]))
    if row["result"] == "H":
        ah, aa = 1.0, 0.0
        history[ht].append(3); history[at].append(0)
        h2h[key].append(ht)
    elif row["result"] == "A":
        ah, aa = 0.0, 1.0
        history[ht].append(0); history[at].append(3)
        h2h[key].append(at)
    else:
        ah, aa = 0.5, 0.5
        history[ht].append(1); history[at].append(1)
        h2h[key].append("D")
    elo[ht] += K * (ah - exp_h)
    elo[at] += K * (aa - exp_a)
    goals_for[ht].append(hg); goals_against[ht].append(ag)
    goals_for[at].append(ag); goals_against[at].append(hg)
    last_match[ht] = row["date"]; last_match[at] = row["date"]
    gd_hist[ht].append(hg - ag); gd_hist[at].append(ag - hg)

# ── 4. Afficher l'état actuel des deux équipes ────────────────────────────
def team_state(team):
    hist5 = history[team][-5:]
    gf5   = goals_for[team][-5:]
    ga5   = goals_against[team][-5:]
    gd5   = gd_hist[team][-5:]
    print(f"\n{'='*60}")
    print(f"  {team.upper()}")
    print(f"{'='*60}")
    print(f"  ELO actuel          : {elo[team]:.1f}")
    print(f"  Forme (5 derniers)  : {np.mean(hist5):.2f} pts/match  {hist5}")
    print(f"  Buts marqués avg5   : {np.mean(gf5):.2f}  {gf5}")
    print(f"  Buts encaissés avg5 : {np.mean(ga5):.2f}  {ga5}")
    print(f"  Diff. buts avg5     : {np.mean(gd5):.2f}  {gd5}")
    print(f"  Dernier match       : {last_match[team].strftime('%Y-%m-%d')}")
    # Stats avancées
    for stat_key in ROLL_STATS:
        vals = adv_stats[team][stat_key]
        if vals:
            print(f"  {stat_key}_avg5       : {np.mean(vals[-5:]):.2f}")

team_state(HOME_TEAM)
team_state(AWAY_TEAM)

# H2H
h2h_key = tuple(sorted([HOME_TEAM, AWAY_TEAM]))
h2h_recent = h2h[h2h_key][-5:]
h_wins = sum(1 for r in h2h_recent if r == HOME_TEAM)
a_wins = sum(1 for r in h2h_recent if r == AWAY_TEAM)
draws  = sum(1 for r in h2h_recent if r == "D")
print(f"\n{'='*60}")
print(f"  HEAD-TO-HEAD (5 derniers)")
print(f"{'='*60}")
print(f"  {HOME_TEAM} wins : {h_wins}  |  Nuls : {draws}  |  {AWAY_TEAM} wins : {a_wins}")
h_h2h = h_wins / len(h2h_recent) if h2h_recent else 0.33
a_h2h = a_wins / len(h2h_recent) if h2h_recent else 0.33

# Jours de repos
h_rest = min((MATCH_DATE - last_match[HOME_TEAM].tz_localize(None) if last_match[HOME_TEAM].tzinfo else MATCH_DATE - last_match[HOME_TEAM]).days, 30)
a_rest = min((MATCH_DATE - last_match[AWAY_TEAM].tz_localize(None) if last_match[AWAY_TEAM].tzinfo else MATCH_DATE - last_match[AWAY_TEAM]).days, 30)
print(f"\n  Repos {HOME_TEAM} : {h_rest} jours (dernier match : {last_match[HOME_TEAM].strftime('%Y-%m-%d')})")
print(f"  Repos {AWAY_TEAM} : {a_rest} jours (dernier match : {last_match[AWAY_TEAM].strftime('%Y-%m-%d')})")

# ── 5. Construire le vecteur de features ──────────────────────────────────
def avg5(lst):
    vals = [v for v in lst[-5:] if v is not None]
    return round(np.mean(vals), 3) if vals else np.nan

match_features = {
    # Baseline (v2)
    "home_elo":                round(elo[HOME_TEAM], 1),
    "away_elo":                round(elo[AWAY_TEAM], 1),
    "elo_diff":                round(elo[HOME_TEAM] - elo[AWAY_TEAM], 1),
    "home_form_5":             round(np.mean(history[HOME_TEAM][-5:]), 3),
    "away_form_5":             round(np.mean(history[AWAY_TEAM][-5:]), 3),
    "home_goals_for_avg":      round(np.mean(goals_for[HOME_TEAM][-5:]), 3),
    "away_goals_for_avg":      round(np.mean(goals_for[AWAY_TEAM][-5:]), 3),
    "home_goals_against_avg":  round(np.mean(goals_against[HOME_TEAM][-5:]), 3),
    "away_goals_against_avg":  round(np.mean(goals_against[AWAY_TEAM][-5:]), 3),
    "home_rest_days":          h_rest,
    "away_rest_days":          a_rest,
    "home_h2h_win_rate":       round(h_h2h, 3),
    "away_h2h_win_rate":       round(a_h2h, 3),
    "home_gd_form":            round(np.mean(gd_hist[HOME_TEAM][-5:]), 3),
    "away_gd_form":            round(np.mean(gd_hist[AWAY_TEAM][-5:]), 3),
    # Europe (v3)
    "home_played_europe":      0,
    "away_played_europe":      0,
}

# Rolling advanced stats
for stat_key in ROLL_STATS:
    match_features[f"home_{stat_key}_avg5"] = avg5(adv_stats[HOME_TEAM][stat_key])
    match_features[f"away_{stat_key}_avg5"] = avg5(adv_stats[AWAY_TEAM][stat_key])

print(f"\n{'='*60}")
print(f"  VECTEUR DE FEATURES : {HOME_TEAM} (D) vs {AWAY_TEAM} (E)")
print(f"{'='*60}")
for k, v in match_features.items():
    print(f"  {k:>35} : {v}")

# ── 6. Charger le modèle v4 et prédire ───────────────────────────────────
with open(MODEL_PKL, "rb") as f:
    model = pickle.load(f)

# Ordre exact extrait du modèle (model.feature_names_in_) :
feature_order = list(model.feature_names_in_)

X = pd.DataFrame([match_features])[feature_order]
proba = model.predict_proba(X)[0]

p_home, p_draw, p_away = proba[0], proba[1], proba[2]
outcomes = [f"{HOME_TEAM} Win", "Nul", f"{AWAY_TEAM} Win"]
most_likely = outcomes[np.argmax(proba)]

print(f"\n{'='*60}")
print(f"  PRÉDICTION — {HOME_TEAM} vs {AWAY_TEAM}")
print(f"  Modèle : XGBoost v4 (stats avancées)")
print(f"  Date   : {MATCH_DATE.strftime('%d %B %Y')}")
print(f"{'='*60}")
print(f"\n  P(Victoire {HOME_TEAM}) : {p_home:.4f}  ({p_home*100:.1f}%)")
print(f"  P(Nul)               : {p_draw:.4f}  ({p_draw*100:.1f}%)")
print(f"  P(Victoire {AWAY_TEAM})   : {p_away:.4f}  ({p_away*100:.1f}%)")
print(f"\n  Cotes implicites :")
print(f"    {HOME_TEAM} : {1/p_home:.2f}")
print(f"    Nul        : {1/p_draw:.2f}")
print(f"    {AWAY_TEAM}   : {1/p_away:.2f}")
print(f"\n  Résultat le plus probable : {most_likely} ({max(proba)*100:.1f}%)")

# Couverture des stats avancées
adv_coverage = sum(1 for s in ROLL_STATS if not np.isnan(match_features.get(f"home_{s}_avg5", np.nan)))
print(f"\n  Couverture stats avancées : {adv_coverage}/{len(ROLL_STATS)} features disponibles")
print(f"  (NaN → XGBoost utilise les features de base pour ces features)")
