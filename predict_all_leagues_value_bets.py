"""
Prédiction XGBoost v4 — 5 Ligues Majeures — Analyse Value Bets vs Betclic
49 matchs : PL J30, La Liga J28, Serie A J29, Bundesliga J26, Ligue 1 J26
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Configuration ──────────────────────────────────────────────────────────
MODEL_PATH   = "/home/ubuntu/football_pred/models/xgb_football_model_v4_advanced.pkl"
DATASET_PATH = "/home/ubuntu/football_pred/dataset/football_matches_v4.csv"
MATCH_DATE   = "2026-03-15 12:00:00"  # Approximate date for next matchday

# Team name mapping: Betclic name -> Dataset name
TEAM_MAP = {
    # Premier League
    "Burnley": "Burnley", "Bournemouth": "Bournemouth",
    "Sunderland": "Sunderland", "Brighton": "Brighton",
    "Chelsea": "Chelsea", "Newcastle": "Newcastle",
    "Arsenal": "Arsenal", "Everton": "Everton",
    "West Ham": "West Ham", "Manchester City": "Manchester City",
    "Manchester United": "Manchester United", "Aston Villa": "Aston Villa",
    "Nottingham Forest": "Nottingham Forest", "Fulham": "Fulham",
    "Crystal Palace": "Crystal Palace", "Leeds": "Leeds",
    "Liverpool": "Liverpool", "Tottenham": "Tottenham",
    "Brentford": "Brentford", "Wolverhampton": "Wolves",
    # La Liga
    "Alaves": "Alaves", "Villarreal": "Villarreal",
    "Girone": "Girona", "Athletic Bilbao": "Athletic Club",
    "Atletico Madrid": "Atletico Madrid", "Getafe": "Getafe",
    "Real Oviedo": "Oviedo", "Valencia": "Valencia",
    "Real Madrid": "Real Madrid", "Elche": "Elche",
    "Majorque": "Mallorca", "Espanyol": "Espanyol",
    "Barcelona": "Barcelona", "Seville": "Sevilla",
    "Betis": "Real Betis", "Celta Vigo": "Celta Vigo",
    "Real Sociedad": "Real Sociedad", "Osasuna": "Osasuna",
    "Rayo Vallecano": "Rayo Vallecano", "Levante": "Levante",
    # Serie A
    "Torino": "Torino", "Parme": "Parma",
    "Inter Milan": "Inter", "Atalanta": "Atalanta",
    "Naples": "Napoli", "Lecce": "Lecce",
    "Udinese": "Udinese", "Juventus": "Juventus",
    "Hellas Verone": "Hellas Verona", "Genoa": "Genoa",
    "Pise": "Pisa", "Cagliari": "Cagliari",
    "Sassuolo": "Sassuolo", "Bologne": "Bologna",
    "Come": "Como", "Roma": "AS Roma",
    "Lazio": "Lazio", "Milan AC": "AC Milan",
    "Cremonese": "Cremonese", "Fiorentina": "Fiorentina",
    # Bundesliga
    "B. Monchengladbach": "Borussia Monchengladbach",
    "St. Pauli": "FC St. Pauli",
    "B. Leverkusen": "Bayer Leverkusen", "Bayern Munich": "Bayern Munich",
    "Francfort": "Eintracht Frankfurt", "Heidenheim": "1. FC Heidenheim",
    "Dortmund": "Borussia Dortmund", "Augsbourg": "FC Augsburg",
    "Hoffenheim": "1899 Hoffenheim", "Wolfsburg": "VfL Wolfsburg",
    "Hamburger SV": "Hamburger SV", "Cologne": "1. FC Köln",
    "Werder Breme": "Werder Bremen", "Mayence": "FSV Mainz 05",
    "Fribourg": "SC Freiburg", "Union Berlin": "Union Berlin",
    "Stuttgart": "VfB Stuttgart", "RB Leipzig": "RB Leipzig",
    # Ligue 1
    "Marseille": "Marseille", "Auxerre": "Auxerre",
    "Lorient": "Lorient", "Lens": "Lens",
    "Angers": "Angers", "Nice": "Nice",
    "Monaco": "Monaco", "Brest": "Stade Brestois 29",
    "Strasbourg": "Strasbourg", "Paris FC": "Paris FC",
    "Le Havre": "Le Havre", "Lyon": "Lyon",
    "Metz": "Metz", "Toulouse": "Toulouse",
    "Rennes": "Rennes", "Lille": "Lille",
}

# All matches organized by league
LEAGUES = {
    "Premier League J30": [
        {"home": "Burnley", "away": "Bournemouth", "odds_h": 3.95, "odds_d": 3.73, "odds_a": 1.82},
        {"home": "Sunderland", "away": "Brighton", "odds_h": 3.28, "odds_d": 3.30, "odds_a": 2.17},
        {"home": "Chelsea", "away": "Newcastle", "odds_h": 1.75, "odds_d": 4.10, "odds_a": 3.93},
        {"home": "Arsenal", "away": "Everton", "odds_h": 1.34, "odds_d": 4.85, "odds_a": 8.50},
        {"home": "West Ham", "away": "Manchester City", "odds_h": 4.50, "odds_d": 4.25, "odds_a": 1.64},
        {"home": "Manchester United", "away": "Aston Villa", "odds_h": 1.73, "odds_d": 3.93, "odds_a": 4.20},
        {"home": "Nottingham Forest", "away": "Fulham", "odds_h": 2.12, "odds_d": 3.40, "odds_a": 3.27},
        {"home": "Crystal Palace", "away": "Leeds", "odds_h": 2.30, "odds_d": 3.27, "odds_a": 3.05},
        {"home": "Liverpool", "away": "Tottenham", "odds_h": 1.32, "odds_d": 5.50, "odds_a": 7.75},
        {"home": "Brentford", "away": "Wolverhampton", "odds_h": 1.54, "odds_d": 4.15, "odds_a": 5.50},
    ],
    "La Liga J28": [
        {"home": "Alaves", "away": "Villarreal", "odds_h": 3.28, "odds_d": 3.40, "odds_a": 2.12},
        {"home": "Girone", "away": "Athletic Bilbao", "odds_h": 2.72, "odds_d": 3.27, "odds_a": 2.53},
        {"home": "Atletico Madrid", "away": "Getafe", "odds_h": 1.58, "odds_d": 3.58, "odds_a": 6.40},
        {"home": "Real Oviedo", "away": "Valencia", "odds_h": 2.85, "odds_d": 3.10, "odds_a": 2.52},
        {"home": "Real Madrid", "away": "Elche", "odds_h": 1.22, "odds_d": 6.50, "odds_a": 10.50},
        {"home": "Majorque", "away": "Espanyol", "odds_h": 2.32, "odds_d": 3.08, "odds_a": 3.18},
        {"home": "Barcelona", "away": "Seville", "odds_h": 1.21, "odds_d": 6.90, "odds_a": 10.00},
        {"home": "Betis", "away": "Celta Vigo", "odds_h": 2.17, "odds_d": 3.23, "odds_a": 3.35},
        {"home": "Real Sociedad", "away": "Osasuna", "odds_h": 1.91, "odds_d": 3.40, "odds_a": 3.95},
        {"home": "Rayo Vallecano", "away": "Levante", "odds_h": 1.69, "odds_d": 3.73, "odds_a": 4.75},
    ],
    "Serie A J29": [
        {"home": "Torino", "away": "Parme", "odds_h": 2.12, "odds_d": 2.95, "odds_a": 3.88},
        {"home": "Inter Milan", "away": "Atalanta", "odds_h": 1.52, "odds_d": 4.25, "odds_a": 5.60},
        {"home": "Naples", "away": "Lecce", "odds_h": 1.36, "odds_d": 4.40, "odds_a": 9.25},
        {"home": "Udinese", "away": "Juventus", "odds_h": 5.60, "odds_d": 3.88, "odds_a": 1.57},
        {"home": "Hellas Verone", "away": "Genoa", "odds_h": 3.18, "odds_d": 3.00, "odds_a": 2.37},
        {"home": "Pise", "away": "Cagliari", "odds_h": 2.45, "odds_d": 3.03, "odds_a": 3.00},
        {"home": "Sassuolo", "away": "Bologne", "odds_h": 2.68, "odds_d": 3.27, "odds_a": 2.57},
        {"home": "Come", "away": "Roma", "odds_h": 2.02, "odds_d": 3.30, "odds_a": 3.32},
        {"home": "Lazio", "away": "Milan AC", "odds_h": 3.80, "odds_d": 3.30, "odds_a": 1.98},
        {"home": "Cremonese", "away": "Fiorentina", "odds_h": 3.78, "odds_d": 3.53, "odds_a": 1.92},
    ],
    "Bundesliga J26": [
        {"home": "B. Monchengladbach", "away": "St. Pauli", "odds_h": 1.95, "odds_d": 3.35, "odds_a": 3.88},
        {"home": "B. Leverkusen", "away": "Bayern Munich", "odds_h": 4.70, "odds_d": 4.50, "odds_a": 1.58},
        {"home": "Francfort", "away": "Heidenheim", "odds_h": 1.50, "odds_d": 4.55, "odds_a": 5.50},
        {"home": "Dortmund", "away": "Augsbourg", "odds_h": 1.39, "odds_d": 5.00, "odds_a": 6.75},
        {"home": "Hoffenheim", "away": "Wolfsburg", "odds_h": 1.45, "odds_d": 4.85, "odds_a": 5.75},
        {"home": "Hamburger SV", "away": "Cologne", "odds_h": 2.05, "odds_d": 3.53, "odds_a": 3.35},
        {"home": "Werder Breme", "away": "Mayence", "odds_h": 2.23, "odds_d": 3.48, "odds_a": 3.00},
        {"home": "Fribourg", "away": "Union Berlin", "odds_h": 2.12, "odds_d": 3.27, "odds_a": 3.43},
        {"home": "Stuttgart", "away": "RB Leipzig", "odds_h": 2.20, "odds_d": 3.93, "odds_a": 2.77},
    ],
    "Ligue 1 J26": [
        {"home": "Marseille", "away": "Auxerre", "odds_h": 1.44, "odds_d": 4.55, "odds_a": 6.40},
        {"home": "Lorient", "away": "Lens", "odds_h": 4.15, "odds_d": 3.63, "odds_a": 1.81},
        {"home": "Angers", "away": "Nice", "odds_h": 3.10, "odds_d": 3.23, "odds_a": 2.28},
        {"home": "Monaco", "away": "Brest", "odds_h": 1.62, "odds_d": 4.10, "odds_a": 4.75},
        {"home": "Strasbourg", "away": "Paris FC", "odds_h": 1.71, "odds_d": 3.80, "odds_a": 4.50},
        {"home": "Le Havre", "away": "Lyon", "odds_h": 3.80, "odds_d": 3.42, "odds_a": 1.94},
        {"home": "Metz", "away": "Toulouse", "odds_h": 3.85, "odds_d": 3.30, "odds_a": 1.97},
        {"home": "Rennes", "away": "Lille", "odds_h": 2.32, "odds_d": 3.37, "odds_a": 2.93},
    ],
}

# ─── Load model & data ──────────────────────────────────────────────────────
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
model_features = model.get_booster().feature_names

df = pd.read_csv(DATASET_PATH)
df['date'] = pd.to_datetime(df['date'])

# ─── Helpers ────────────────────────────────────────────────────────────────
def get_last(df, team, before_date):
    mask = ((df['home_team'] == team) | (df['away_team'] == team)) & (df['date'] < before_date)
    sub  = df[mask].sort_values('date', ascending=False)
    return sub.iloc[0] if len(sub) else None

def pick(row, team, home_col, away_col):
    if row is None:
        return np.nan
    return row[home_col] if row['home_team'] == team else row[away_col]

def build_features(df, home_team, away_team, match_date):
    h = get_last(df, home_team, match_date)
    a = get_last(df, away_team, match_date)

    if h is None or a is None:
        return None

    match_dt = pd.to_datetime(match_date).tz_localize(None)
    h_dt = pd.to_datetime(h['date'])
    a_dt = pd.to_datetime(a['date'])
    if hasattr(h_dt, 'tz') and h_dt.tz is not None:
        h_dt = h_dt.tz_localize(None)
    if hasattr(a_dt, 'tz') and a_dt.tz is not None:
        a_dt = a_dt.tz_localize(None)

    feat = {
        'home_elo':               pick(h, home_team, 'home_elo', 'away_elo'),
        'away_elo':               pick(a, away_team, 'home_elo', 'away_elo'),
        'elo_diff':               pick(h, home_team, 'home_elo', 'away_elo') - pick(a, away_team, 'home_elo', 'away_elo'),
        'home_form_5':            pick(h, home_team, 'home_form_5', 'away_form_5'),
        'away_form_5':            pick(a, away_team, 'home_form_5', 'away_form_5'),
        'home_goals_for_avg':     pick(h, home_team, 'home_goals_for_avg', 'away_goals_for_avg'),
        'away_goals_for_avg':     pick(a, away_team, 'home_goals_for_avg', 'away_goals_for_avg'),
        'home_goals_against_avg': pick(h, home_team, 'home_goals_against_avg', 'away_goals_against_avg'),
        'away_goals_against_avg': pick(a, away_team, 'home_goals_against_avg', 'away_goals_against_avg'),
        'home_h2h_win_rate':      pick(h, home_team, 'home_h2h_win_rate', 'away_h2h_win_rate'),
        'away_h2h_win_rate':      pick(a, away_team, 'home_h2h_win_rate', 'away_h2h_win_rate'),
        'home_gd_form':           pick(h, home_team, 'home_gd_form', 'away_gd_form'),
        'away_gd_form':           pick(a, away_team, 'home_gd_form', 'away_gd_form'),
        'home_played_europe':     pick(h, home_team, 'home_played_europe', 'away_played_europe'),
        'away_played_europe':     pick(a, away_team, 'home_played_europe', 'away_played_europe'),
        'home_rest_days':         (match_dt - h_dt).days,
        'away_rest_days':         (match_dt - a_dt).days,
    }

    adv_cols = [
        'shots_on_goal_avg5', 'total_shots_avg5', 'shots_insidebox_avg5',
        'ball_possession_avg5', 'total_passes_avg5', 'passes_pct_avg5',
        'corner_kicks_avg5', 'fouls_avg5', 'expected_goals_avg5'
    ]
    for col in adv_cols:
        feat[f'home_{col}'] = pick(h, home_team, f'home_{col}', f'away_{col}')
        feat[f'away_{col}'] = pick(a, away_team, f'home_{col}', f'away_{col}')

    return feat

# ─── Compute predictions for all matches ────────────────────────────────────
all_results = []
skipped = []

for league_name, matches in LEAGUES.items():
    print(f"\n{'='*80}")
    print(f"  {league_name}")
    print(f"{'='*80}")

    for m in matches:
        home_ds = TEAM_MAP.get(m['home'], m['home'])
        away_ds = TEAM_MAP.get(m['away'], m['away'])

        feat = build_features(df, home_ds, away_ds, MATCH_DATE)
        if feat is None:
            print(f"  SKIP: {m['home']:25s} vs {m['away']:25s} — données manquantes")
            skipped.append(f"{m['home']} vs {m['away']}")
            continue

        X = pd.DataFrame([feat])[model_features]
        probs = model.predict_proba(X)[0]  # [H, D, A]

        # Betclic implied probabilities (adjusted for margin)
        raw_implied = [1/m['odds_h'], 1/m['odds_d'], 1/m['odds_a']]
        margin = sum(raw_implied) - 1.0
        implied = [p / sum(raw_implied) for p in raw_implied]

        print(f"  {m['home']:25s} vs {m['away']:25s} -> H={probs[0]*100:5.1f}% D={probs[1]*100:5.1f}% A={probs[2]*100:5.1f}%  (margin: {margin*100:.1f}%)")

        for i, outcome in enumerate(['H', 'D', 'A']):
            odds = [m['odds_h'], m['odds_d'], m['odds_a']][i]
            edge = probs[i] - implied[i]
            ev = (probs[i] * odds) - 1
            all_results.append({
                'League': league_name,
                'Match': f"{m['home']} vs {m['away']}",
                'Outcome': outcome,
                'Model Prob': probs[i],
                'Betclic Odds': odds,
                'Implied Prob': implied[i],
                'Edge': edge,
                'EV': ev,
                'Value Bet': edge > 0.05,
            })

results_df = pd.DataFrame(all_results)

# ─── Print detailed tables per league ────────────────────────────────────────
for league_name in LEAGUES.keys():
    league_df = results_df[results_df['League'] == league_name]
    matches_in_league = league_df['Match'].unique()

    print(f"\n{'='*100}")
    print(f"  {league_name} — Détail des prédictions")
    print(f"{'='*100}")

    for match_name in matches_in_league:
        sub = league_df[league_df['Match'] == match_name]
        print(f"\n  {match_name}")
        print(f"  {'Outcome':>8s}  {'Model%':>8s}  {'Implied%':>8s}  {'Odds':>6s}  {'Edge':>8s}  {'EV':>8s}  {'Value?':>7s}")
        for _, row in sub.iterrows():
            flag = "  YES" if row['Value Bet'] else ""
            print(f"  {row['Outcome']:>8s}  {row['Model Prob']*100:7.1f}%  {row['Implied Prob']*100:7.1f}%  {row['Betclic Odds']:6.2f}  {row['Edge']*100:+7.1f}%  {row['EV']*100:+7.1f}%  {flag}")

# ─── Value bets summary ─────────────────────────────────────────────────────
vb = results_df[results_df['Value Bet']].sort_values('Edge', ascending=False)
print(f"\n{'='*120}")
print(f"  RÉSUMÉ — TOUS LES VALUE BETS (Edge > 5%) — triés par edge décroissant")
print(f"{'='*120}")
if len(vb) == 0:
    print("  Aucun value bet identifié avec un edge > 5%.")
else:
    print(f"  {'#':>3s}  {'League':<20s}  {'Match':<35s}  {'Out':>4s}  {'Model%':>8s}  {'Implied%':>8s}  {'Odds':>6s}  {'Edge':>8s}  {'EV':>8s}")
    for rank, (_, row) in enumerate(vb.iterrows(), 1):
        print(f"  {rank:3d}  {row['League']:<20s}  {row['Match']:<35s}  {row['Outcome']:>4s}  {row['Model Prob']*100:7.1f}%  {row['Implied Prob']*100:7.1f}%  {row['Betclic Odds']:6.2f}  {row['Edge']*100:+7.1f}%  {row['EV']*100:+7.1f}%")

print(f"\nTotal: {len(vb)} value bets sur {len(results_df)} résultats ({len(results_df)//3} matchs)")
if skipped:
    print(f"Matchs ignorés (données manquantes): {', '.join(skipped)}")

# ─── Save CSV ────────────────────────────────────────────────────────────────
results_df.to_csv('/home/ubuntu/all_leagues_value_bets_v4.csv', index=False)
vb.to_csv('/home/ubuntu/value_bets_summary_v4.csv', index=False)

# ─── Visualization ──────────────────────────────────────────────────────────
# Chart 1: Value bets edge chart
if len(vb) > 0:
    fig, ax = plt.subplots(figsize=(14, max(6, len(vb) * 0.5)))
    labels = [f"{r['Match']} ({r['Outcome']})" for _, r in vb.iterrows()]
    edges = [r['Edge'] * 100 for _, r in vb.iterrows()]
    evs = [r['EV'] * 100 for _, r in vb.iterrows()]

    # Color by league
    league_colors = {
        "Premier League J30": "#3D195B",
        "La Liga J28": "#FF4B44",
        "Serie A J29": "#024494",
        "Bundesliga J26": "#D20515",
        "Ligue 1 J26": "#091C3E",
    }
    colors = [league_colors.get(r['League'], '#333') for _, r in vb.iterrows()]

    bars = ax.barh(range(len(labels)), edges, color=colors, alpha=0.85, height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Edge (%) = Prob. Modèle - Prob. Implicite Betclic', fontsize=11)
    ax.set_title('Value Bets — XGBoost v4 vs Cotes Betclic\n5 Ligues Majeures', fontsize=14, fontweight='bold')
    ax.axvline(x=5, color='green', linestyle='--', alpha=0.6, label='Seuil 5%')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add EV annotations
    for i, (e, ev) in enumerate(zip(edges, evs)):
        ax.text(e + 0.3, i, f'EV: {ev:+.1f}%', va='center', fontsize=8, color='#333')

    # Legend for leagues
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in league_colors.items()]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig('/home/ubuntu/all_leagues_value_bets_v4.png', dpi=150)
    plt.close()
    print(f"\nGraphique sauvegardé : /home/ubuntu/all_leagues_value_bets_v4.png")

# Chart 2: Overview of all matches — model vs implied for each league
fig, axes = plt.subplots(5, 1, figsize=(18, 30))
fig.suptitle('Prédictions XGBoost v4 vs Cotes Betclic — 5 Ligues Majeures',
             fontsize=16, fontweight='bold', y=0.995)

for idx, (league_name, ax) in enumerate(zip(LEAGUES.keys(), axes)):
    league_df = results_df[results_df['League'] == league_name]
    match_names = league_df['Match'].unique()
    n = len(match_names)
    x = np.arange(n)
    width = 0.12

    outcomes = ['H', 'D', 'A']
    colors_model   = ['#1565C0', '#616161', '#C62828']
    colors_implied = ['#64B5F6', '#BDBDBD', '#EF9A9A']

    for i, (outcome, cm, ci) in enumerate(zip(outcomes, colors_model, colors_implied)):
        model_vals = []
        implied_vals = []
        for mn in match_names:
            sub = league_df[(league_df['Match'] == mn) & (league_df['Outcome'] == outcome)]
            model_vals.append(sub['Model Prob'].values[0] * 100)
            implied_vals.append(sub['Implied Prob'].values[0] * 100)

        ax.bar(x + (i*2)*width - 2.5*width, model_vals, width, color=cm, alpha=0.9)
        ax.bar(x + (i*2+1)*width - 2.5*width, implied_vals, width, color=ci, alpha=0.7)

    # Shorten match names for display
    short_names = []
    for mn in match_names:
        parts = mn.split(' vs ')
        h = parts[0][:12]
        a = parts[1][:12]
        short_names.append(f"{h}\nvs\n{a}")

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=7)
    ax.set_ylabel('Probabilité (%)', fontsize=9)
    ax.set_title(league_name, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if idx == 0:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1565C0', label='H Modèle'), Patch(facecolor='#64B5F6', label='H Betclic'),
            Patch(facecolor='#616161', label='D Modèle'), Patch(facecolor='#BDBDBD', label='D Betclic'),
            Patch(facecolor='#C62828', label='A Modèle'), Patch(facecolor='#EF9A9A', label='A Betclic'),
        ]
        ax.legend(handles=legend_elements, fontsize=7, ncol=6, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('/home/ubuntu/all_leagues_overview_v4.png', dpi=150)
plt.close()
print(f"Graphique overview sauvegardé : /home/ubuntu/all_leagues_overview_v4.png")

# ─── Generate Markdown report ────────────────────────────────────────────────
md = []
md.append("# Analyse Value Bets — XGBoost v4 vs Cotes Betclic")
md.append("## 5 Ligues Majeures — Prochaine Journée (Mars 2026)\n")
md.append(f"**Date de l'analyse** : 10 mars 2026\n")
md.append(f"**Modèle** : XGBoost v4 (35 features, incluant statistiques avancées)\n")
md.append(f"**Nombre de matchs analysés** : {len(results_df)//3}\n")
if skipped:
    md.append(f"**Matchs ignorés** : {', '.join(skipped)}\n")
md.append("---\n")

# Value bets summary first
md.append("## Value Bets Identifiés (Edge > 5%)\n")
if len(vb) > 0:
    md.append("| # | Ligue | Match | Résultat | Prob. Modèle | Prob. Betclic | Cote | Edge | EV |")
    md.append("|:-:|:------|:------|:--------:|:---:|:---:|:---:|:---:|:---:|")
    for rank, (_, row) in enumerate(vb.iterrows(), 1):
        md.append(f"| {rank} | {row['League']} | {row['Match']} | {row['Outcome']} | {row['Model Prob']*100:.1f}% | {row['Implied Prob']*100:.1f}% | {row['Betclic Odds']:.2f} | {row['Edge']*100:+.1f}% | {row['EV']*100:+.1f}% |")
else:
    md.append("Aucun value bet identifié avec un edge > 5%.\n")

md.append("\n---\n")

# Detail per league
for league_name in LEAGUES.keys():
    league_df = results_df[results_df['League'] == league_name]
    match_names = league_df['Match'].unique()

    md.append(f"## {league_name}\n")
    md.append("| Match | H (Modèle) | H (Betclic) | D (Modèle) | D (Betclic) | A (Modèle) | A (Betclic) | Value Bet |")
    md.append("|:------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for mn in match_names:
        sub = league_df[league_df['Match'] == mn]
        h = sub[sub['Outcome'] == 'H'].iloc[0]
        d = sub[sub['Outcome'] == 'D'].iloc[0]
        a = sub[sub['Outcome'] == 'A'].iloc[0]

        # Find value bets for this match
        vb_flags = []
        for _, r in sub.iterrows():
            if r['Value Bet']:
                vb_flags.append(f"{r['Outcome']} ({r['Edge']*100:+.1f}%)")
        vb_str = ', '.join(vb_flags) if vb_flags else '-'

        md.append(f"| {mn} | {h['Model Prob']*100:.1f}% | {h['Implied Prob']*100:.1f}% | {d['Model Prob']*100:.1f}% | {d['Implied Prob']*100:.1f}% | {a['Model Prob']*100:.1f}% | {a['Implied Prob']*100:.1f}% | {vb_str} |")

    md.append("")

md_text = '\n'.join(md)
with open('/home/ubuntu/all_leagues_value_bets_v4.md', 'w') as f:
    f.write(md_text)
print(f"Rapport Markdown sauvegardé : /home/ubuntu/all_leagues_value_bets_v4.md")
print("\nTERMINÉ.")
