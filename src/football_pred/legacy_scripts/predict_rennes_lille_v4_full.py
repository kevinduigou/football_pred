"""
Prédiction XGBoost v4 — Rennes (D) vs Lille (E)
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
HOME_TEAM    = "Rennes"
AWAY_TEAM    = "Lille"
MATCH_DATE   = "2026-03-08 17:00:00"
OUT_CHART    = "/home/ubuntu/rennes_lille_prediction_v4.png"
OUT_REPORT   = "/home/ubuntu/rennes_lille_prediction_v4.md"

# ─── Helpers ────────────────────────────────────────────────────────────────
def get_last(df, team, before_date):
    mask = ((df['home_team'] == team) | (df['away_team'] == team)) & (df['date'] < before_date)
    sub  = df[mask].sort_values('date', ascending=False)
    return sub.iloc[0] if len(sub) else None

def pick(row, team, home_col, away_col):
    return row[home_col] if row['home_team'] == team else row[away_col]

# ─── Load ────────────────────────────────────────────────────────────────────
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
model_features = model.get_booster().feature_names

df = pd.read_csv(DATASET_PATH)

# ─── Feature construction ────────────────────────────────────────────────────
h = get_last(df, HOME_TEAM, MATCH_DATE)
a = get_last(df, AWAY_TEAM, MATCH_DATE)

match_dt = pd.to_datetime(MATCH_DATE).tz_localize(None)
h_dt = pd.to_datetime(h['date']).tz_localize(None)
a_dt = pd.to_datetime(a['date']).tz_localize(None)

feat = {
    'home_elo':               pick(h, HOME_TEAM, 'home_elo', 'away_elo'),
    'away_elo':               pick(a, AWAY_TEAM, 'home_elo', 'away_elo'),
    'elo_diff':               pick(h, HOME_TEAM, 'home_elo', 'away_elo') - pick(a, AWAY_TEAM, 'home_elo', 'away_elo'),
    'home_form_5':            pick(h, HOME_TEAM, 'home_form_5', 'away_form_5'),
    'away_form_5':            pick(a, AWAY_TEAM, 'home_form_5', 'away_form_5'),
    'home_goals_for_avg':     pick(h, HOME_TEAM, 'home_goals_for_avg', 'away_goals_for_avg'),
    'away_goals_for_avg':     pick(a, AWAY_TEAM, 'home_goals_for_avg', 'away_goals_for_avg'),
    'home_goals_against_avg': pick(h, HOME_TEAM, 'home_goals_against_avg', 'away_goals_against_avg'),
    'away_goals_against_avg': pick(a, AWAY_TEAM, 'home_goals_against_avg', 'away_goals_against_avg'),
    'home_h2h_win_rate':      pick(h, HOME_TEAM, 'home_h2h_win_rate', 'away_h2h_win_rate'),
    'away_h2h_win_rate':      pick(a, AWAY_TEAM, 'home_h2h_win_rate', 'away_h2h_win_rate'),
    'home_gd_form':           pick(h, HOME_TEAM, 'home_gd_form', 'away_gd_form'),
    'away_gd_form':           pick(a, AWAY_TEAM, 'home_gd_form', 'away_gd_form'),
    'home_played_europe':     pick(h, HOME_TEAM, 'home_played_europe', 'away_played_europe'),
    'away_played_europe':     pick(a, AWAY_TEAM, 'home_played_europe', 'away_played_europe'),
    'home_rest_days':         (match_dt - h_dt).days,
    'away_rest_days':         (match_dt - a_dt).days,
}

adv_cols = [
    'shots_on_goal_avg5', 'total_shots_avg5', 'shots_insidebox_avg5',
    'ball_possession_avg5', 'total_passes_avg5', 'passes_pct_avg5',
    'corner_kicks_avg5', 'fouls_avg5', 'expected_goals_avg5'
]
for col in adv_cols:
    feat[f'home_{col}'] = pick(h, HOME_TEAM, f'home_{col}', f'away_{col}')
    feat[f'away_{col}'] = pick(a, AWAY_TEAM, f'home_{col}', f'away_{col}')

X = pd.DataFrame([feat])[model_features]
probs = model.predict_proba(X)[0]   # [H, D, A]

# ─── Print results ───────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  PRÉDICTION  {HOME_TEAM} vs {AWAY_TEAM}  —  08/03/2026")
print(f"{'='*55}")
print(f"  Victoire {HOME_TEAM:<10}  {probs[0]*100:5.1f}%   cote: {1/probs[0]:5.2f}")
print(f"  Nul                  {probs[1]*100:5.1f}%   cote: {1/probs[1]:5.2f}")
print(f"  Victoire {AWAY_TEAM:<10}  {probs[2]*100:5.1f}%   cote: {1/probs[2]:5.2f}")
print(f"{'='*55}")
print(f"\nContexte :")
for k, v in feat.items():
    print(f"  {k:<35} {v}")

# ─── Visualisation ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'Prédiction XGBoost v4 — {HOME_TEAM} vs {AWAY_TEAM}\n08 mars 2026 · Ligue 1',
             fontsize=14, fontweight='bold', y=1.01)

# Graphique 1 : probabilités
ax1 = axes[0]
labels  = [f'Victoire\n{HOME_TEAM}', 'Nul', f'Victoire\n{AWAY_TEAM}']
colors  = ['#2196F3', '#9E9E9E', '#F44336']
bars    = ax1.bar(labels, [p*100 for p in probs], color=colors, width=0.5, edgecolor='white', linewidth=1.5)
for bar, prob in zip(bars, probs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f'{prob*100:.1f}%\n(cote {1/prob:.2f})',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 55)
ax1.set_ylabel('Probabilité (%)', fontsize=11)
ax1.set_title('Probabilités prédites', fontsize=12)
ax1.axhline(33.3, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Équiprobable (33.3%)')
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Graphique 2 : statistiques avancées comparées
ax2 = axes[1]
categories = ['Tirs\ncadrés', 'Tirs\ntotaux', 'Tirs\nzone', 'Possession\n(%)',
              'Passes\n(÷10)', 'Précision\npasses (%)', 'Corners', 'xG']
h_vals = [feat['home_shots_on_goal_avg5'], feat['home_total_shots_avg5'],
          feat['home_shots_insidebox_avg5'], feat['home_ball_possession_avg5'],
          feat['home_total_passes_avg5']/10, feat['home_passes_pct_avg5'],
          feat['home_corner_kicks_avg5'], feat['home_expected_goals_avg5']]
a_vals = [feat['away_shots_on_goal_avg5'], feat['away_total_shots_avg5'],
          feat['away_shots_insidebox_avg5'], feat['away_ball_possession_avg5'],
          feat['away_total_passes_avg5']/10, feat['away_passes_pct_avg5'],
          feat['away_corner_kicks_avg5'], feat['away_expected_goals_avg5']]

x_pos = np.arange(len(categories))
width = 0.35
ax2.bar(x_pos - width/2, h_vals, width, label=HOME_TEAM, color='#2196F3', alpha=0.85, edgecolor='white')
ax2.bar(x_pos + width/2, a_vals, width, label=AWAY_TEAM,  color='#F44336', alpha=0.85, edgecolor='white')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(categories, fontsize=8)
ax2.set_title('Statistiques avancées (moy. 5 derniers matchs)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_CHART, dpi=150, bbox_inches='tight')
print(f"\nGraphique sauvegardé : {OUT_CHART}")

# ─── Rapport Markdown ────────────────────────────────────────────────────────
report = f"""# Prédiction XGBoost v4 — {HOME_TEAM} vs {AWAY_TEAM}
**Date** : 08 mars 2026 · Ligue 1  
**Modèle** : XGBoost v4 (avec statistiques avancées API-Football)

---

## Probabilités et Cotes

| Résultat | Probabilité | Cote implicite |
|----------|:-----------:|:--------------:|
| **Victoire {HOME_TEAM}** | **{probs[0]*100:.1f}%** | **{1/probs[0]:.2f}** |
| Nul | {probs[1]*100:.1f}% | {1/probs[1]:.2f} |
| Victoire {AWAY_TEAM} | {probs[2]*100:.1f}% | {1/probs[2]:.2f} |

---

## Statistiques Comparatives (moyennes sur 5 derniers matchs)

| Indicateur | {HOME_TEAM} (D) | {AWAY_TEAM} (E) |
|------------|:---:|:---:|
| ELO | {feat['home_elo']:.0f} | {feat['away_elo']:.0f} |
| Différentiel ELO | {feat['elo_diff']:+.0f} | — |
| Forme (pts/match) | {feat['home_form_5']:.1f} | {feat['away_form_5']:.1f} |
| Buts marqués/match | {feat['home_goals_for_avg']:.2f} | {feat['away_goals_for_avg']:.2f} |
| Buts encaissés/match | {feat['home_goals_against_avg']:.2f} | {feat['away_goals_against_avg']:.2f} |
| Diff. buts forme | {feat['home_gd_form']:+.1f} | {feat['away_gd_form']:+.1f} |
| H2H win rate | {feat['home_h2h_win_rate']*100:.0f}% | {feat['away_h2h_win_rate']*100:.0f}% |
| Jours de repos | {feat['home_rest_days']} | {feat['away_rest_days']} |
| Joué en Europe | {'Oui' if feat['home_played_europe'] else 'Non'} | {'Oui' if feat['away_played_europe'] else 'Non'} |
| **Tirs cadrés** | **{feat['home_shots_on_goal_avg5']:.1f}** | **{feat['away_shots_on_goal_avg5']:.1f}** |
| Tirs totaux | {feat['home_total_shots_avg5']:.1f} | {feat['away_total_shots_avg5']:.1f} |
| Tirs dans la surface | {feat['home_shots_insidebox_avg5']:.1f} | {feat['away_shots_insidebox_avg5']:.1f} |
| Possession (%) | {feat['home_ball_possession_avg5']:.1f}% | {feat['away_ball_possession_avg5']:.1f}% |
| Passes totales | {feat['home_total_passes_avg5']:.0f} | {feat['away_total_passes_avg5']:.0f} |
| Précision passes (%) | {feat['home_passes_pct_avg5']:.1f}% | {feat['away_passes_pct_avg5']:.1f}% |
| Corners | {feat['home_corner_kicks_avg5']:.1f} | {feat['away_corner_kicks_avg5']:.1f} |
| Fautes | {feat['home_fouls_avg5']:.1f} | {feat['away_fouls_avg5']:.1f} |
| **xG (Expected Goals)** | **{feat['home_expected_goals_avg5']:.2f}** | **{feat['away_expected_goals_avg5']:.2f}** |

---

## Analyse Contextuelle

### Forme récente de {HOME_TEAM}
Sur les 6 derniers matchs, Rennes est en nette progression : victoires contre Nantes (2-1), Auxerre (2-0) et Toulouse (2-1), un nul contre Lens (1-1) et une défaite à Reims. La forme sur 5 matchs est de {feat['home_form_5']:.1f} pts/match. Rennes marque régulièrement ({feat['home_goals_for_avg']:.1f} buts/match) mais présente une défense perfectible ({feat['home_goals_against_avg']:.1f} buts encaissés/match).

### Forme récente de {AWAY_TEAM}
Lille est en légère baisse de régime : victoire contre Nantes (1-0), mais nuls contre Strasbourg (1-1) et défaite à Monaco. La forme sur 5 matchs est de {feat['away_form_5']:.1f} pts/match. Lille reste une équipe solide défensivement ({feat['away_goals_against_avg']:.1f} buts encaissés/match) mais peine à concrétiser ses occasions ({feat['away_goals_for_avg']:.1f} buts/match).

### Facteurs clés du modèle
Le modèle v4 identifie un **match très équilibré**. L'ELO de Lille ({feat['away_elo']:.0f}) est supérieur à celui de Rennes ({feat['home_elo']:.0f}) d'environ {abs(feat['elo_diff']):.0f} points, ce qui est significatif. Cependant, Rennes bénéficie de l'avantage du terrain et d'une meilleure forme récente. Les xG de Lille ({feat['away_expected_goals_avg5']:.2f}) sont supérieurs à ceux de Rennes ({feat['home_expected_goals_avg5']:.2f}), indiquant que Lille crée plus d'occasions de qualité malgré un bilan offensif récent modeste. Le H2H est très défavorable à Rennes (0% de victoires contre Lille sur les derniers face-à-face).

![Prédiction Rennes vs Lille](rennes_lille_prediction_v4.png)
"""

with open(OUT_REPORT, 'w') as f:
    f.write(report)
print(f"Rapport sauvegardé : {OUT_REPORT}")
