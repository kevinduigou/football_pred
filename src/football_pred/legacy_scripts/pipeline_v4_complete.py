"""
Complete pipeline: build advanced stats CSV, rolling averages, train v2/v3/v4, compare.
Uses all data currently in the cache (~4800 matches).
"""

import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    log_loss, classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────
DATASET   = "/home/ubuntu/football_pred/dataset/football_matches.csv"
CACHE     = "/home/ubuntu/football_pred/dataset/advanced_stats_cache.json"
OUT_CSV   = "/home/ubuntu/football_pred/dataset/football_matches_v4.csv"
STATS_CSV = "/home/ubuntu/football_pred/dataset/advanced_stats.csv"
RES_DIR   = "/home/ubuntu/football_pred/results"
MDL_DIR   = "/home/ubuntu/football_pred/models"
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(MDL_DIR, exist_ok=True)

ROLLING_W = 5

# Stats we want rolling averages for
ROLL_STATS = [
    "shots_on_goal", "total_shots", "shots_insidebox",
    "ball_possession", "total_passes", "passes_pct",
    "corner_kicks", "fouls", "expected_goals",
]

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 – Build advanced_stats.csv from cache
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Build advanced_stats.csv from cache")
print("=" * 70)

with open(CACHE) as f:
    cache = json.load(f)
print(f"Cache entries: {len(cache)}")

df = pd.read_csv(DATASET)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"Dataset: {df.shape}")

RAW_STATS = [
    "Shots on Goal", "Shots off Goal", "Total Shots", "Blocked Shots",
    "Shots insidebox", "Shots outsidebox", "Fouls", "Corner Kicks",
    "Offsides", "Ball Possession", "Yellow Cards", "Red Cards",
    "Goalkeeper Saves", "Total passes", "Passes accurate", "Passes %",
    "expected_goals",
]

def clean(n):
    return n.lower().replace(" ", "_").replace("%", "pct")

rows = []
for _, row in df.iterrows():
    fid = str(int(row["fixture_id"]))
    rd = {"fixture_id": int(fid)}
    entry = cache.get(fid, {})
    if entry and len(entry) >= 2:
        tids = list(entry.keys())
        t1, t2 = entry[tids[0]], entry[tids[1]]
        n1, n2 = t1.get("team_name", ""), t2.get("team_name", "")
        dh, da = row["home_team"].lower(), row["away_team"].lower()
        # Match home/away
        if n1.lower() in dh or dh in n1.lower():
            hs, aws = t1, t2
        elif n2.lower() in dh or dh in n2.lower():
            hs, aws = t2, t1
        else:
            hs, aws = t1, t2  # default: API returns home first
        for sn in RAW_STATS:
            cn = clean(sn)
            rd[f"home_{cn}"] = hs.get(sn)
            rd[f"away_{cn}"] = aws.get(sn)
    else:
        for sn in RAW_STATS:
            cn = clean(sn)
            rd[f"home_{cn}"] = None
            rd[f"away_{cn}"] = None
    rows.append(rd)

sdf = pd.DataFrame(rows)
sdf.to_csv(STATS_CSV, index=False)
matched = sdf["home_shots_on_goal"].notna().sum()
xg_count = sdf["home_expected_goals"].notna().sum()
print(f"Matched: {matched}, With xG: {xg_count}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 – Build rolling averages (vectorized, fast)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Build rolling averages")
print("=" * 70)

# Merge raw stats into main df
df = df.merge(sdf, on="fixture_id", how="left")

# Build team-centric records
home_records = df[["fixture_id", "date", "home_team"]].copy()
home_records.columns = ["fixture_id", "date", "team"]
for s in ROLL_STATS:
    home_records[s] = df.get(f"home_{s}")

away_records = df[["fixture_id", "date", "away_team"]].copy()
away_records.columns = ["fixture_id", "date", "team"]
for s in ROLL_STATS:
    away_records[s] = df.get(f"away_{s}")

team_df = pd.concat([home_records, away_records], ignore_index=True)
team_df = team_df.sort_values("date").reset_index(drop=True)
print(f"Team-match records: {team_df.shape}")

# Convert stat columns to numeric (handle '39%' -> 39.0, None -> NaN)
for s in ROLL_STATS:
    team_df[s] = team_df[s].astype(str).str.replace('%', '', regex=False)
    team_df[s] = pd.to_numeric(team_df[s], errors='coerce')

# Compute rolling averages per team (shift to avoid leakage)
for s in ROLL_STATS:
    col = f"{s}_avg{ROLLING_W}"
    team_df[col] = (
        team_df.groupby("team")[s]
        .transform(lambda x: x.shift(1).rolling(window=ROLLING_W, min_periods=1).mean())
    )

# Create lookup dict: (fixture_id, team) -> rolling stats
team_df["fid_team"] = team_df["fixture_id"].astype(str) + "_" + team_df["team"]
roll_cols = [f"{s}_avg{ROLLING_W}" for s in ROLL_STATS]
lookup = team_df.set_index("fid_team")[roll_cols].to_dict("index")

# Map back to main df
for s in ROLL_STATS:
    col = f"{s}_avg{ROLLING_W}"
    home_col = f"home_{col}"
    away_col = f"away_{col}"
    home_keys = df["fixture_id"].astype(int).astype(str) + "_" + df["home_team"]
    away_keys = df["fixture_id"].astype(int).astype(str) + "_" + df["away_team"]
    df[home_col] = home_keys.map(lambda k: lookup.get(k, {}).get(col))
    df[away_col] = away_keys.map(lambda k: lookup.get(k, {}).get(col))

# Drop raw per-match stat columns
raw_drop = [c for c in df.columns
            if (c.startswith("home_") or c.startswith("away_"))
            and "_avg" not in c
            and c.split("_", 1)[1] in [clean(s) for s in RAW_STATS]]
df = df.drop(columns=raw_drop, errors="ignore")

# Coverage summary
new_cols = [f"home_{s}_avg{ROLLING_W}" for s in ROLL_STATS] + \
           [f"away_{s}_avg{ROLLING_W}" for s in ROLL_STATS]
print("\nCoverage of rolling features:")
for c in new_cols[:4]:  # show a few
    nn = df[c].notna().sum()
    print(f"  {c}: {nn}/{len(df)} ({nn/len(df)*100:.1f}%)")
print("  ...")

df.to_csv(OUT_CSV, index=False)
print(f"\nSaved enriched dataset: {df.shape} -> {OUT_CSV}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 – Train and compare v2, v3, v4
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Train and compare models")
print("=" * 70)

BASELINE_FEATURES = [
    "home_elo", "away_elo", "elo_diff",
    "home_form_5", "away_form_5",
    "home_goals_for_avg", "away_goals_for_avg",
    "home_goals_against_avg", "away_goals_against_avg",
    "home_rest_days", "away_rest_days",
    "home_h2h_win_rate", "away_h2h_win_rate",
    "home_gd_form", "away_gd_form",
]

EUROPE_FEATURES = BASELINE_FEATURES + [
    "home_played_europe", "away_played_europe",
]

# Exclude xG if zero coverage
xg_home = f"home_expected_goals_avg{ROLLING_W}"
xg_away = f"away_expected_goals_avg{ROLLING_W}"
xg_coverage = df[xg_home].notna().sum()
ADVANCED_ROLLING = []
for s in ROLL_STATS:
    if s == "expected_goals" and xg_coverage == 0:
        continue
    ADVANCED_ROLLING.append(f"home_{s}_avg{ROLLING_W}")
    ADVANCED_ROLLING.append(f"away_{s}_avg{ROLLING_W}")

ADVANCED_FEATURES = EUROPE_FEATURES + ADVANCED_ROLLING
print(f"Advanced features: {len(ADVANCED_FEATURES)} ({len(ADVANCED_ROLLING)} new rolling)")
print(f"xG coverage: {xg_coverage} ({'included' if xg_coverage > 0 else 'excluded'})")

y = df["result"].map({"H": 0, "D": 1, "A": 2})

# Chronological split 80/20
n = len(df)
split_idx = int(n * 0.8)
y_train_full = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"\nTrain: {split_idx}, Test: {n - split_idx}")
print(f"Test seasons: {sorted(df.iloc[split_idx:]['season'].unique())}")

# Check advanced coverage in test set
test_adv = df.iloc[split_idx:][ADVANCED_ROLLING[0]].notna().sum()
print(f"Test set advanced stats coverage: {test_adv}/{n - split_idx} ({test_adv/(n-split_idx)*100:.1f}%)")

model_configs = [
    ("baseline_v2", BASELINE_FEATURES),
    ("europe_v3", EUROPE_FEATURES),
    ("advanced_v4", ADVANCED_FEATURES),
]

results_summary = {}

for model_name, feature_cols in model_configs:
    print(f"\n{'─' * 65}")
    print(f"  MODEL: {model_name} ({len(feature_cols)} features)")
    print(f"{'─' * 65}")

    X = df[feature_cols]
    X_train_full = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    # Grid search
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [4, 5, 6],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.7, 0.8],
        "min_child_weight": [3, 5],
    }

    xgb_base = XGBClassifier(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42, tree_method="hist",
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(xgb_base, param_grid, cv=cv, scoring="neg_log_loss",
                      n_jobs=-1, verbose=0, refit=True)
    gs.fit(X_train_full, y_train_full)

    best = gs.best_params_
    cv_ll = -gs.best_score_
    print(f"  Best params: {best}")
    print(f"  CV Log Loss: {cv_ll:.4f}")

    # Final model with eval set
    xgb_final = XGBClassifier(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42, tree_method="hist",
        **best,
    )
    xgb_final.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)],
                  verbose=False)

    proba = xgb_final.predict_proba(X_test)
    pred = xgb_final.predict(X_test)
    t_ll = log_loss(y_test, proba)
    t_acc = accuracy_score(y_test, pred)
    t_f1w = f1_score(y_test, pred, average="weighted")
    t_f1m = f1_score(y_test, pred, average="macro")

    report = classification_report(y_test, pred, target_names=["Home", "Draw", "Away"], output_dict=True)
    print(f"  Test Log Loss: {t_ll:.4f}")
    print(f"  Test Accuracy: {t_acc:.4f}")
    print(f"  Test F1 Weighted: {t_f1w:.4f}")
    print(f"  Test F1 Macro: {t_f1m:.4f}")
    print(classification_report(y_test, pred, target_names=["Home", "Draw", "Away"]))

    results_summary[model_name] = {
        "features": len(feature_cols),
        "feature_names": feature_cols,
        "best_params": best,
        "cv_logloss": round(cv_ll, 4),
        "test_logloss": round(t_ll, 4),
        "test_accuracy": round(t_acc, 4),
        "test_f1_weighted": round(t_f1w, 4),
        "test_f1_macro": round(t_f1m, 4),
        "report": report,
    }

    # ── Confusion matrix ──
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Home", "Draw", "Away"],
                yticklabels=["Home", "Draw", "Away"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    plt.savefig(f"{RES_DIR}/confusion_matrix_{model_name}.png", dpi=150)
    plt.close()

    # ── Feature importance ──
    imp = xgb_final.feature_importances_
    fi = pd.DataFrame({"feature": feature_cols, "importance": imp}).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(8, len(feature_cols) * 0.35)))
    colors = []
    for fn in fi["feature"]:
        if fn in ADVANCED_ROLLING:
            colors.append("#FF6F00")
        elif fn in ["home_played_europe", "away_played_europe"]:
            colors.append("#E53935")
        else:
            colors.append("#1565C0")
    ax.barh(fi["feature"], fi["importance"], color=colors)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(f"Feature Importance – {model_name}")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#1565C0", label="Baseline (v2)"),
        Patch(facecolor="#E53935", label="Europe (v3)"),
        Patch(facecolor="#FF6F00", label="Advanced Stats (v4)"),
    ], loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{RES_DIR}/feature_importance_{model_name}.png", dpi=150)
    plt.close()

    # ── Training curves ──
    ev = xgb_final.evals_result()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ev["validation_0"]["mlogloss"], label="Train", alpha=0.8)
    ax.plot(ev["validation_1"]["mlogloss"], label="Validation", alpha=0.8)
    ax.set_xlabel("Boosting Round"); ax.set_ylabel("Log Loss")
    ax.set_title(f"Training Curves – {model_name}")
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{RES_DIR}/training_curves_{model_name}.png", dpi=150)
    plt.close()

    # ── Test predictions ──
    tr = df.iloc[split_idx:][["date", "league_name", "home_team", "away_team", "result"]].copy().reset_index(drop=True)
    tr["P_Home"] = proba[:, 0].round(4)
    tr["P_Draw"] = proba[:, 1].round(4)
    tr["P_Away"] = proba[:, 2].round(4)
    tr["predicted"] = pd.Series(pred).map({0: "H", 1: "D", 2: "A"})
    tr["correct"] = tr["result"] == tr["predicted"]
    tr.to_csv(f"{RES_DIR}/test_predictions_{model_name}.csv", index=False)

    # Save v4 production model
    if model_name == "advanced_v4":
        xgb_prod = XGBClassifier(
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=42, tree_method="hist",
            **best,
        )
        xgb_prod.fit(X, y, verbose=False)
        xgb_prod.save_model(f"{MDL_DIR}/xgb_football_model_v4_advanced.json")
        with open(f"{MDL_DIR}/xgb_football_model_v4_advanced.pkl", "wb") as f_pkl:
            pickle.dump(xgb_prod, f_pkl)
        print(f"  Production model saved.")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 – Comparison summary & charts
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Comparison summary")
print("=" * 70)

b = results_summary["baseline_v2"]
e = results_summary["europe_v3"]
a = results_summary["advanced_v4"]

comp = pd.DataFrame({
    "Metric": ["Features", "CV Log Loss", "Test Log Loss", "Test Accuracy",
               "Test F1 (Weighted)", "Test F1 (Macro)",
               "Home Win F1", "Draw F1", "Away Win F1"],
    "Baseline (v2)": [
        b["features"], b["cv_logloss"], b["test_logloss"],
        b["test_accuracy"], b["test_f1_weighted"], b["test_f1_macro"],
        round(b["report"]["Home"]["f1-score"], 4),
        round(b["report"]["Draw"]["f1-score"], 4),
        round(b["report"]["Away"]["f1-score"], 4),
    ],
    "Europe (v3)": [
        e["features"], e["cv_logloss"], e["test_logloss"],
        e["test_accuracy"], e["test_f1_weighted"], e["test_f1_macro"],
        round(e["report"]["Home"]["f1-score"], 4),
        round(e["report"]["Draw"]["f1-score"], 4),
        round(e["report"]["Away"]["f1-score"], 4),
    ],
    "Advanced (v4)": [
        a["features"], a["cv_logloss"], a["test_logloss"],
        a["test_accuracy"], a["test_f1_weighted"], a["test_f1_macro"],
        round(a["report"]["Home"]["f1-score"], 4),
        round(a["report"]["Draw"]["f1-score"], 4),
        round(a["report"]["Away"]["f1-score"], 4),
    ],
})
print(comp.to_string(index=False))
comp.to_csv(f"{RES_DIR}/comparison_v2_v3_v4.csv", index=False)

with open(f"{RES_DIR}/results_summary_v4.json", "w") as f:
    json.dump(results_summary, f, indent=2, default=str)

# ── Comparison chart ──
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
models = ["Baseline (v2)", "Europe (v3)", "Advanced (v4)"]
colors = ["#42A5F5", "#E53935", "#FF6F00"]

metrics_k = ["test_accuracy", "test_f1_weighted", "test_f1_macro"]
labels_k = ["Accuracy", "F1 (Weighted)", "F1 (Macro)"]
vals = {m: [results_summary[k][mk] for mk in metrics_k]
        for m, k in zip(models, results_summary.keys())}

x = np.arange(len(labels_k))
w = 0.25
for i, (m, c) in enumerate(zip(models, colors)):
    axes[0].bar(x + i * w, vals[m], w, label=m, color=c)
    for j, v in enumerate(vals[m]):
        axes[0].text(j + i * w, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
axes[0].set_ylabel("Score"); axes[0].set_title("Accuracy & F1 Scores")
axes[0].set_xticks(x + w); axes[0].set_xticklabels(labels_k); axes[0].legend()
axes[0].set_ylim(0, max(max(v) for v in vals.values()) * 1.15)

ll_vals = [b["test_logloss"], e["test_logloss"], a["test_logloss"]]
bars = axes[1].bar(models, ll_vals, color=colors)
axes[1].set_ylabel("Log Loss (lower is better)"); axes[1].set_title("Test Log Loss")
axes[1].set_ylim(min(ll_vals) * 0.95, max(ll_vals) * 1.05)
for bar, v in zip(bars, ll_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.001, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=9)

classes = ["Home", "Draw", "Away"]
for i, (m, k, c) in enumerate(zip(models, results_summary.keys(), colors)):
    r = results_summary[k]["report"]
    f1v = [r[cl]["f1-score"] for cl in classes]
    axes[2].bar(np.arange(len(classes)) + i * w, f1v, w, label=m, color=c)
axes[2].set_ylabel("F1 Score"); axes[2].set_title("Per-class F1 Scores")
axes[2].set_xticks(np.arange(len(classes)) + w)
axes[2].set_xticklabels(classes); axes[2].legend()

plt.tight_layout()
plt.savefig(f"{RES_DIR}/comparison_chart_v4.png", dpi=150)
plt.close()

print(f"\nAll results saved to {RES_DIR}/")
print("PIPELINE COMPLETE.")
