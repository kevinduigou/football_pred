"""
Prédiction XGBoost v4 — Ligue 1 J26 — Analyse Value Bets vs Betclic
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
MATCH_DATE   = "2026-03-08 12:00:00"

MATCHES = [
    {"home": "Marseille",          "away": "Auxerre",   "odds_h": 1.44, "odds_d": 4.60, "odds_a": 6.50},
    {"home": "Lorient",            "away": "Lens",      "odds_h": 4.15, "odds_d": 3.63, "odds_a": 1.81},
    {"home": "Angers",             "away": "Nice",      "odds_h": 3.10, "odds_d": 3.23, "odds_a": 2.28},
    {"home": "Monaco",             "away": "Stade Brestois 29", "odds_h": 1.62, "odds_d": 4.10, "odds_a": 4.75},
    {"home": "Strasbourg",         "away": "Paris FC",  "odds_h": 1.71, "odds_d": 3.80, "odds_a": 4.50},
    {"home": "Le Havre",           "away": "Lyon",      "odds_h": 3.80, "odds_d": 3.42, "odds_a": 1.94},
    {"home": "Metz",               "away": "Toulouse",  "odds_h": 3.83, "odds_d": 3.30, "odds_a": 1.97},
    {"home": "Rennes",             "away": "Lille",     "odds_h": 2.37, "odds_d": 3.38, "odds_a": 2.85},
]

# ─── Load model & data ──────────────────────────────────────────────────────
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
model_features = model.get_booster().feature_names

df = pd.read_csv(DATASET_PATH)

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
    h_dt = pd.to_datetime(h['date']).tz_localize(None) if pd.notna(h['date']) else match_dt
    a_dt = pd.to_datetime(a['date']).tz_localize(None) if pd.notna(a['date']) else match_dt

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

# ─── Compute predictions for all 8 matches ──────────────────────────────────
results = []
for m in MATCHES:
    feat = build_features(df, m['home'], m['away'], MATCH_DATE)
    if feat is None:
        print(f"SKIP: {m['home']} vs {m['away']} — données manquantes")
        continue

    X = pd.DataFrame([feat])[model_features]
    probs = model.predict_proba(X)[0]  # [H, D, A]

    # Betclic implied probabilities (adjusted for margin)
    raw_implied = [1/m['odds_h'], 1/m['odds_d'], 1/m['odds_a']]
    margin = sum(raw_implied) - 1.0
    implied = [p / sum(raw_implied) for p in raw_implied]

    # Edge and EV for each outcome
    for i, outcome in enumerate(['H', 'D', 'A']):
        odds = [m['odds_h'], m['odds_d'], m['odds_a']][i]
        edge = probs[i] - implied[i]
        ev = (probs[i] * odds) - 1
        results.append({
            'Match': f"{m['home']} vs {m['away']}",
            'Outcome': outcome,
            'Model Prob': probs[i],
            'Betclic Odds': odds,
            'Implied Prob (adj)': implied[i],
            'Edge': edge,
            'EV': ev,
            'Value Bet': edge > 0.05,
        })

    print(f"OK: {m['home']:15s} vs {m['away']:20s} -> H={probs[0]*100:.1f}% D={probs[1]*100:.1f}% A={probs[2]*100:.1f}%")

results_df = pd.DataFrame(results)

# ─── Print full comparison table ─────────────────────────────────────────────
print("\n" + "="*100)
print("TABLEAU COMPLET — Prédictions XGBoost v4 vs Cotes Betclic — Ligue 1 J26")
print("="*100)
for m in MATCHES:
    match_name = f"{m['home']} vs {m['away']}"
    sub = results_df[results_df['Match'] == match_name]
    print(f"\n{match_name}")
    print(f"  {'Outcome':>8s}  {'Model%':>8s}  {'Implied%':>8s}  {'Odds':>6s}  {'Edge':>8s}  {'EV':>8s}  {'Value?':>7s}")
    for _, row in sub.iterrows():
        flag = "  YES" if row['Value Bet'] else ""
        print(f"  {row['Outcome']:>8s}  {row['Model Prob']*100:7.1f}%  {row['Implied Prob (adj)']*100:7.1f}%  {row['Betclic Odds']:6.2f}  {row['Edge']*100:+7.1f}%  {row['EV']*100:+7.1f}%  {flag}")

# ─── Value bets sorted by edge ───────────────────────────────────────────────
vb = results_df[results_df['Value Bet']].sort_values('Edge', ascending=False)
print("\n" + "="*100)
print("VALUE BETS (Edge > 5%) — triés par edge décroissant")
print("="*100)
if len(vb) == 0:
    print("  Aucun value bet identifié avec un edge > 5%.")
else:
    print(f"  {'Match':<30s}  {'Outcome':>8s}  {'Model%':>8s}  {'Implied%':>8s}  {'Odds':>6s}  {'Edge':>8s}  {'EV':>8s}")
    for _, row in vb.iterrows():
        print(f"  {row['Match']:<30s}  {row['Outcome']:>8s}  {row['Model Prob']*100:7.1f}%  {row['Implied Prob (adj)']*100:7.1f}%  {row['Betclic Odds']:6.2f}  {row['Edge']*100:+7.1f}%  {row['EV']*100:+7.1f}%")

# ─── Save CSV ────────────────────────────────────────────────────────────────
results_df.to_csv('/home/ubuntu/j26_value_bets_v4.csv', index=False)
print(f"\nCSV sauvegardé : /home/ubuntu/j26_value_bets_v4.csv")

# ─── Visualisation ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 12))
fig.suptitle('Ligue 1 J26 — XGBoost v4 vs Cotes Betclic\nAnalyse Value Bets',
             fontsize=15, fontweight='bold', y=0.98)

# Chart 1: Grouped bar chart — Model prob vs Implied prob for each match/outcome
ax1 = axes[0]
match_names_short = []
for m in MATCHES:
    h_short = m['home'].replace('Stade Brestois 29', 'Brest')
    a_short = m['away'].replace('Stade Brestois 29', 'Brest')
    match_names_short.append(f"{h_short}\nvs\n{a_short}")

n_matches = len(MATCHES)
x = np.arange(n_matches)
width = 0.12

outcomes = ['H', 'D', 'A']
colors_model   = ['#1565C0', '#616161', '#C62828']
colors_implied = ['#64B5F6', '#BDBDBD', '#EF9A9A']

for i, (outcome, cm, ci) in enumerate(zip(outcomes, colors_model, colors_implied)):
    sub = results_df[results_df['Outcome'] == outcome]
    model_vals   = [sub[sub['Match'].str.startswith(m['home'])]['Model Prob'].values[0]*100 for m in MATCHES]
    implied_vals = [sub[sub['Match'].str.startswith(m['home'])]['Implied Prob (adj)'].values[0]*100 for m in MATCHES]

    ax1.bar(x + (i*2)*width - 2.5*width, model_vals, width, color=cm, label=f'{outcome} Modèle' if i == 0 else f'{outcome} Modèle')
    ax1.bar(x + (i*2+1)*width - 2.5*width, implied_vals, width, color=ci, label=f'{outcome} Betclic' if i == 0 else f'{outcome} Betclic')

ax1.set_xticks(x)
ax1.set_xticklabels(match_names_short, fontsize=8)
ax1.set_ylabel('Probabilité (%)', fontsize=11)
ax1.set_title('Probabilités : Modèle XGBoost v4 (foncé) vs Betclic implicite (clair)', fontsize=12)
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1565C0', label='H Modèle'), Patch(facecolor='#64B5F6', label='H Betclic'),
    Patch(facecolor='#616161', label='D Modèle'), Patch(facecolor='#BDBDBD', label='D Betclic'),
    Patch(facecolor='#C62828', label='A Modèle'), Patch(facecolor='#EF9A9A', label='A Betclic'),
]
ax1.legend(handles=legend_elements, fontsize=8, ncol=6, loc='upper right')

# Chart 2: Edge chart — horizontal bars for all outcomes, highlight value bets
ax2 = axes[1]
# Sort by edge
sorted_df = results_df.sort_values('Edge', ascending=True)
labels = [f"{row['Match']} ({row['Outcome']})" for _, row in sorted_df.iterrows()]
edges  = sorted_df['Edge'].values * 100
colors_edge = ['#2E7D32' if e > 5 else ('#81C784' if e > 0 else '#EF5350') for e in edges]

bars = ax2.barh(range(len(labels)), edges, color=colors_edge, edgecolor='white', linewidth=0.5)
ax2.set_yticks(range(len(labels)))
ax2.set_yticklabels(labels, fontsize=7)
ax2.set_xlabel('Edge (%) = Prob. Modèle - Prob. Implicite Betclic', fontsize=11)
ax2.set_title('Edge par résultat — Vert foncé = Value Bet (edge > 5%)', fontsize=12)
ax2.axvline(0, color='black', linewidth=0.8)
ax2.axvline(5, color='green', linewidth=1, linestyle='--', alpha=0.7, label='Seuil value bet (5%)')
ax2.axvline(-5, color='red', linewidth=1, linestyle='--', alpha=0.7)
ax2.legend(fontsize=9)
ax2.grid(axis='x', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/home/ubuntu/j26_value_bets_v4.png', dpi=150, bbox_inches='tight')
print(f"Graphique sauvegardé : /home/ubuntu/j26_value_bets_v4.png")

# ─── Markdown report ─────────────────────────────────────────────────────────
md = """# Analyse Value Bets — Ligue 1 J26 — XGBoost v4 vs Betclic
**Date** : 08 mars 2026  
**Modèle** : XGBoost v4 (avec statistiques avancées API-Football)  
**Bookmaker** : Betclic

---

## Méthodologie

Les probabilités implicites des cotes Betclic sont calculées en ajustant pour la marge du bookmaker :

> **Prob. implicite ajustée** = (1 / cote) / somme(1 / cotes)

L'**edge** est défini comme la différence entre la probabilité du modèle et la probabilité implicite ajustée. L'**Expected Value (EV)** est calculée comme `(prob_modèle × cote) - 1`. Un pari est considéré comme un **value bet** si l'edge dépasse 5%.

---

## Tableau Complet des Prédictions

"""

# Build the full table
md += "| Match | Résultat | Prob. Modèle | Prob. Betclic | Cote | Edge | EV | Value Bet |\n"
md += "|-------|:--------:|:------------:|:-------------:|:----:|:----:|:--:|:---------:|\n"
for _, row in results_df.iterrows():
    vb_flag = "**OUI**" if row['Value Bet'] else "Non"
    md += f"| {row['Match']} | {row['Outcome']} | {row['Model Prob']*100:.1f}% | {row['Implied Prob (adj)']*100:.1f}% | {row['Betclic Odds']:.2f} | {row['Edge']*100:+.1f}% | {row['EV']*100:+.1f}% | {vb_flag} |\n"

md += "\n---\n\n## Value Bets Identifiés (Edge > 5%)\n\n"

vb = results_df[results_df['Value Bet']].sort_values('Edge', ascending=False)
if len(vb) == 0:
    md += "> Aucun value bet identifié avec un edge supérieur à 5%.\n"
else:
    md += "| Rang | Match | Résultat | Prob. Modèle | Prob. Betclic | Cote | Edge | EV |\n"
    md += "|:----:|-------|:--------:|:------------:|:-------------:|:----:|:----:|:--:|\n"
    for rank, (_, row) in enumerate(vb.iterrows(), 1):
        md += f"| {rank} | {row['Match']} | {row['Outcome']} | {row['Model Prob']*100:.1f}% | {row['Implied Prob (adj)']*100:.1f}% | {row['Betclic Odds']:.2f} | {row['Edge']*100:+.1f}% | {row['EV']*100:+.1f}% |\n"

md += """
---

## Résumé par Match

"""

for m in MATCHES:
    match_name = f"{m['home']} vs {m['away']}"
    sub = results_df[results_df['Match'] == match_name]
    h_row = sub[sub['Outcome'] == 'H'].iloc[0]
    d_row = sub[sub['Outcome'] == 'D'].iloc[0]
    a_row = sub[sub['Outcome'] == 'A'].iloc[0]

    best = sub.loc[sub['Model Prob'].idxmax()]
    h_short = m['home'].replace('Stade Brestois 29', 'Brest')
    a_short = m['away'].replace('Stade Brestois 29', 'Brest')

    md += f"### {h_short} vs {a_short}\n\n"
    md += f"| | Victoire {h_short} | Nul | Victoire {a_short} |\n"
    md += f"|---|:---:|:---:|:---:|\n"
    md += f"| **Modèle v4** | {h_row['Model Prob']*100:.1f}% | {d_row['Model Prob']*100:.1f}% | {a_row['Model Prob']*100:.1f}% |\n"
    md += f"| **Betclic** | {h_row['Implied Prob (adj)']*100:.1f}% | {d_row['Implied Prob (adj)']*100:.1f}% | {a_row['Implied Prob (adj)']*100:.1f}% |\n"
    md += f"| **Cote** | {m['odds_h']:.2f} | {m['odds_d']:.2f} | {m['odds_a']:.2f} |\n"
    md += f"| **Edge** | {h_row['Edge']*100:+.1f}% | {d_row['Edge']*100:+.1f}% | {a_row['Edge']*100:+.1f}% |\n\n"

    # Find value bets for this match
    match_vb = sub[sub['Value Bet']]
    if len(match_vb) > 0:
        for _, vrow in match_vb.iterrows():
            outcome_name = {'H': f'Victoire {h_short}', 'D': 'Nul', 'A': f'Victoire {a_short}'}[vrow['Outcome']]
            md += f"> **VALUE BET** : {outcome_name} à {vrow['Betclic Odds']:.2f} (edge {vrow['Edge']*100:+.1f}%, EV {vrow['EV']*100:+.1f}%)\n\n"
    else:
        md += f"> Pas de value bet identifié pour ce match.\n\n"

md += """---

## Visualisations

![Comparaison Modèle vs Betclic](j26_value_bets_v4.png)

---

*Analyse générée par le modèle XGBoost v4 avec statistiques avancées API-Football.*
"""

with open('/home/ubuntu/j26_value_bets_v4.md', 'w') as f:
    f.write(md)
print(f"Rapport sauvegardé : /home/ubuntu/j26_value_bets_v4.md")
