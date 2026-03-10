"""
Train a Random Forest model on football_matches_v4.csv and compare
with the existing XGBoost v4 model.
Same train/test split, same features, same evaluation methodology.
Metrics: Accuracy, Log Loss, F1 Macro, F1 Weighted, RPS, Brier Score, ECE.
"""

import json, os, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    log_loss, classification_report, confusion_matrix,
    accuracy_score, f1_score, brier_score_loss
)
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────
DATASET = "/home/ubuntu/football_pred/dataset/football_matches_v4.csv"
RES_DIR = "/home/ubuntu/football_pred/results"
MDL_DIR = "/home/ubuntu/football_pred/models"
OUT_DIR = "/home/ubuntu"
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(MDL_DIR, exist_ok=True)

ROLLING_W = 5
ROLL_STATS = [
    "shots_on_goal", "total_shots", "shots_insidebox",
    "ball_possession", "total_passes", "passes_pct",
    "corner_kicks", "fouls", "expected_goals",
]

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

ADVANCED_ROLLING = []
for s in ROLL_STATS:
    ADVANCED_ROLLING.append(f"home_{s}_avg{ROLLING_W}")
    ADVANCED_ROLLING.append(f"away_{s}_avg{ROLLING_W}")

ADVANCED_FEATURES = EUROPE_FEATURES + ADVANCED_ROLLING

# ══════════════════════════════════════════════════════════════════════════
# Helper functions for metrics
# ══════════════════════════════════════════════════════════════════════════

def compute_rps(y_true, y_proba):
    """Ranked Probability Score for ordered outcomes (H=0, D=1, A=2)."""
    n = len(y_true)
    rps_sum = 0.0
    for i in range(n):
        actual = np.zeros(3)
        actual[y_true.iloc[i]] = 1.0
        cum_pred = np.cumsum(y_proba[i])
        cum_actual = np.cumsum(actual)
        rps_sum += np.sum((cum_pred - cum_actual) ** 2) / 2.0
    return rps_sum / n


def compute_multiclass_brier(y_true, y_proba):
    """Multi-class Brier score."""
    n = len(y_true)
    bs = 0.0
    for i in range(n):
        actual = np.zeros(3)
        actual[y_true.iloc[i]] = 1.0
        bs += np.sum((y_proba[i] - actual) ** 2)
    return bs / n


def compute_ece(y_true, y_proba, n_bins=10):
    """Expected Calibration Error per class and mean."""
    ece_per_class = {}
    class_names = ["Home", "Draw", "Away"]
    for c in range(3):
        probs = y_proba[:, c]
        actual = (y_true.values == c).astype(float)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(probs)
        for b in range(n_bins):
            mask = (probs >= bin_edges[b]) & (probs < bin_edges[b + 1])
            if b == n_bins - 1:
                mask = (probs >= bin_edges[b]) & (probs <= bin_edges[b + 1])
            n_in_bin = mask.sum()
            if n_in_bin > 0:
                avg_pred = probs[mask].mean()
                avg_actual = actual[mask].mean()
                ece += (n_in_bin / total) * abs(avg_pred - avg_actual)
        ece_per_class[class_names[c]] = round(ece, 5)
    ece_per_class["Mean"] = round(np.mean(list(ece_per_class.values())), 5)
    return ece_per_class


def calibration_data(y_true, y_proba, n_bins=10):
    """Return calibration data for plotting reliability diagrams."""
    class_names = ["Home", "Draw", "Away"]
    cal = {}
    for c in range(3):
        probs = y_proba[:, c]
        actual = (y_true.values == c).astype(float)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_actuals = []
        bin_counts = []
        for b in range(n_bins):
            mask = (probs >= bin_edges[b]) & (probs < bin_edges[b + 1])
            if b == n_bins - 1:
                mask = (probs >= bin_edges[b]) & (probs <= bin_edges[b + 1])
            n_in_bin = mask.sum()
            if n_in_bin > 0:
                bin_centers.append(probs[mask].mean())
                bin_actuals.append(actual[mask].mean())
                bin_counts.append(n_in_bin)
        cal[class_names[c]] = {
            "centers": bin_centers,
            "actuals": bin_actuals,
            "counts": bin_counts,
        }
    return cal


# ══════════════════════════════════════════════════════════════════════════
# Load data and prepare split
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Loading dataset and preparing train/test split")
print("=" * 70)

df = pd.read_csv(DATASET)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"Dataset: {df.shape}")

# Check xG coverage
xg_col = f"home_expected_goals_avg{ROLLING_W}"
xg_cov = df[xg_col].notna().sum()
if xg_cov == 0:
    # Remove xG features if no coverage
    ADVANCED_ROLLING = [c for c in ADVANCED_ROLLING if "expected_goals" not in c]
    ADVANCED_FEATURES = EUROPE_FEATURES + ADVANCED_ROLLING
print(f"xG coverage: {xg_cov} ({'included' if xg_cov > 0 else 'excluded'})")
print(f"Advanced features: {len(ADVANCED_FEATURES)} ({len(ADVANCED_ROLLING)} rolling)")

y = df["result"].map({"H": 0, "D": 1, "A": 2})

# Chronological split 80/20
n = len(df)
split_idx = int(n * 0.8)
y_train_full = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

X_adv = df[ADVANCED_FEATURES]
X_train_full = X_adv.iloc[:split_idx]
X_test = X_adv.iloc[split_idx:]

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"Train: {split_idx}, Test: {n - split_idx}")
print(f"Test seasons: {sorted(df.iloc[split_idx:]['season'].unique())}")
test_adv_cov = df.iloc[split_idx:][ADVANCED_ROLLING[0]].notna().sum()
print(f"Test advanced stats coverage: {test_adv_cov}/{n-split_idx} ({test_adv_cov/(n-split_idx)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Train XGBoost v4 (reproduce existing)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: Train XGBoost v4")
print("=" * 70)

xgb_param_grid = {
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
print("  Running GridSearchCV for XGBoost...")
gs_xgb = GridSearchCV(xgb_base, xgb_param_grid, cv=cv, scoring="neg_log_loss",
                       n_jobs=-1, verbose=0, refit=True)
gs_xgb.fit(X_train_full, y_train_full)

xgb_best_params = gs_xgb.best_params_
xgb_cv_ll = -gs_xgb.best_score_
print(f"  Best params: {xgb_best_params}")
print(f"  CV Log Loss: {xgb_cv_ll:.4f}")

# Train with eval set for training curves
xgb_final = XGBClassifier(
    objective="multi:softprob", num_class=3,
    eval_metric="mlogloss", random_state=42, tree_method="hist",
    **xgb_best_params,
)
xgb_final.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_valid, y_valid)],
              verbose=False)

xgb_proba = gs_xgb.predict_proba(X_test)
xgb_pred = gs_xgb.predict(X_test)

xgb_metrics = {
    "cv_logloss": round(xgb_cv_ll, 4),
    "test_logloss": round(log_loss(y_test, xgb_proba), 4),
    "test_accuracy": round(accuracy_score(y_test, xgb_pred), 4),
    "test_f1_weighted": round(f1_score(y_test, xgb_pred, average="weighted"), 4),
    "test_f1_macro": round(f1_score(y_test, xgb_pred, average="macro"), 4),
    "test_rps": round(compute_rps(y_test, xgb_proba), 5),
    "test_brier": round(compute_multiclass_brier(y_test, xgb_proba), 5),
}
xgb_ece = compute_ece(y_test, xgb_proba)
xgb_metrics.update({f"ece_{k.lower()}": v for k, v in xgb_ece.items()})

print(f"  Test Log Loss: {xgb_metrics['test_logloss']}")
print(f"  Test Accuracy: {xgb_metrics['test_accuracy']}")
print(f"  Test F1 Macro: {xgb_metrics['test_f1_macro']}")
print(f"  Test RPS: {xgb_metrics['test_rps']}")
print(f"  Test Brier: {xgb_metrics['test_brier']}")
print(f"  ECE: {xgb_ece}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Train Random Forest with hyperparameter tuning
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Train Random Forest (with GridSearchCV)")
print("=" * 70)

# Fill NaN for RF (RF doesn't handle NaN natively)
X_train_full_rf = X_train_full.fillna(-999)
X_test_rf = X_test.fillna(-999)
X_train_rf = X_train.fillna(-999)
X_valid_rf = X_valid.fillna(-999)

from sklearn.model_selection import RandomizedSearchCV

rf_param_dist = {
    "n_estimators": [200, 500],
    "max_depth": [10, 15, 20],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
    "max_features": ["sqrt", 0.5],
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

print("  Running RandomizedSearchCV for Random Forest (20 iterations)...")
gs_rf = RandomizedSearchCV(rf_base, rf_param_dist, n_iter=20, cv=cv, 
                           scoring="neg_log_loss", n_jobs=-1, verbose=0, 
                           random_state=42, refit=True)
gs_rf.fit(X_train_full_rf, y_train_full)

rf_best_params = gs_rf.best_params_
rf_cv_ll = -gs_rf.best_score_
print(f"  Best params: {rf_best_params}")
print(f"  CV Log Loss: {rf_cv_ll:.4f}")

rf_proba = gs_rf.predict_proba(X_test_rf)
rf_pred = gs_rf.predict(X_test_rf)

rf_metrics = {
    "cv_logloss": round(rf_cv_ll, 4),
    "test_logloss": round(log_loss(y_test, rf_proba), 4),
    "test_accuracy": round(accuracy_score(y_test, rf_pred), 4),
    "test_f1_weighted": round(f1_score(y_test, rf_pred, average="weighted"), 4),
    "test_f1_macro": round(f1_score(y_test, rf_pred, average="macro"), 4),
    "test_rps": round(compute_rps(y_test, rf_proba), 5),
    "test_brier": round(compute_multiclass_brier(y_test, rf_proba), 5),
}
rf_ece = compute_ece(y_test, rf_proba)
rf_metrics.update({f"ece_{k.lower()}": v for k, v in rf_ece.items()})

print(f"  Test Log Loss: {rf_metrics['test_logloss']}")
print(f"  Test Accuracy: {rf_metrics['test_accuracy']}")
print(f"  Test F1 Macro: {rf_metrics['test_f1_macro']}")
print(f"  Test RPS: {rf_metrics['test_rps']}")
print(f"  Test Brier: {rf_metrics['test_brier']}")
print(f"  ECE: {rf_ece}")

# Save RF model
with open(f"{MDL_DIR}/rf_football_model_v4.pkl", "wb") as f:
    pickle.dump(gs_rf.best_estimator_, f)
print("  RF model saved.")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Comparison table
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Comparison XGBoost v4 vs Random Forest")
print("=" * 70)

comp = pd.DataFrame({
    "Metric": [
        "CV Log Loss", "Test Log Loss", "Test Accuracy",
        "Test F1 (Weighted)", "Test F1 (Macro)",
        "RPS", "Brier Score",
        "ECE Home", "ECE Draw", "ECE Away", "ECE Mean",
    ],
    "XGBoost v4": [
        xgb_metrics["cv_logloss"], xgb_metrics["test_logloss"],
        xgb_metrics["test_accuracy"], xgb_metrics["test_f1_weighted"],
        xgb_metrics["test_f1_macro"], xgb_metrics["test_rps"],
        xgb_metrics["test_brier"],
        xgb_metrics["ece_home"], xgb_metrics["ece_draw"],
        xgb_metrics["ece_away"], xgb_metrics["ece_mean"],
    ],
    "Random Forest": [
        rf_metrics["cv_logloss"], rf_metrics["test_logloss"],
        rf_metrics["test_accuracy"], rf_metrics["test_f1_weighted"],
        rf_metrics["test_f1_macro"], rf_metrics["test_rps"],
        rf_metrics["test_brier"],
        rf_metrics["ece_home"], rf_metrics["ece_draw"],
        rf_metrics["ece_away"], rf_metrics["ece_mean"],
    ],
})

# Add "Better" column
better = []
lower_is_better = ["CV Log Loss", "Test Log Loss", "RPS", "Brier Score",
                    "ECE Home", "ECE Draw", "ECE Away", "ECE Mean"]
for _, row in comp.iterrows():
    xv = row["XGBoost v4"]
    rv = row["Random Forest"]
    if row["Metric"] in lower_is_better:
        better.append("XGBoost" if xv < rv else ("Random Forest" if rv < xv else "Tie"))
    else:
        better.append("XGBoost" if xv > rv else ("Random Forest" if rv > xv else "Tie"))
comp["Winner"] = better

print(comp.to_string(index=False))
comp.to_csv(f"{RES_DIR}/xgboost_vs_rf_comparison.csv", index=False)
comp.to_csv(f"{OUT_DIR}/xgboost_vs_rf_comparison.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Visualizations
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Generating comparison charts")
print("=" * 70)

# --- 4a: Bar chart comparison of all metrics ---
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle("XGBoost v4 vs Random Forest — Comparaison Complète", fontsize=18, fontweight="bold", y=0.98)

colors = {"XGBoost v4": "#1565C0", "Random Forest": "#2E7D32"}

# Accuracy
ax = axes[0, 0]
vals = [xgb_metrics["test_accuracy"], rf_metrics["test_accuracy"]]
bars = ax.bar(["XGBoost v4", "Random Forest"], vals, color=[colors["XGBoost v4"], colors["Random Forest"]], width=0.5)
for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
ax.set_title("Accuracy (↑)", fontsize=13, fontweight="bold")
ax.set_ylim(min(vals) - 0.02, max(vals) + 0.02)
ax.axhline(y=max(vals), color="gray", linestyle="--", alpha=0.3)

# Log Loss
ax = axes[0, 1]
vals = [xgb_metrics["test_logloss"], rf_metrics["test_logloss"]]
bars = ax.bar(["XGBoost v4", "Random Forest"], vals, color=[colors["XGBoost v4"], colors["Random Forest"]], width=0.5)
for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
ax.set_title("Log Loss (↓)", fontsize=13, fontweight="bold")
ax.set_ylim(min(vals) - 0.02, max(vals) + 0.02)

# F1 Scores
ax = axes[0, 2]
x = np.arange(2)
w = 0.3
f1w = [xgb_metrics["test_f1_weighted"], rf_metrics["test_f1_weighted"]]
f1m = [xgb_metrics["test_f1_macro"], rf_metrics["test_f1_macro"]]
bars1 = ax.bar(x - w/2, f1w, w, label="F1 Weighted", color=[colors["XGBoost v4"], colors["Random Forest"]], alpha=0.8)
bars2 = ax.bar(x + w/2, f1m, w, label="F1 Macro", color=[colors["XGBoost v4"], colors["Random Forest"]], alpha=0.5)
for b, v in zip(bars1, f1w): ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
for b, v in zip(bars2, f1m): ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(["XGBoost v4", "Random Forest"])
ax.set_title("F1 Scores (↑)", fontsize=13, fontweight="bold")
ax.legend()
ax.set_ylim(min(f1m) - 0.03, max(f1w) + 0.03)

# RPS
ax = axes[1, 0]
vals = [xgb_metrics["test_rps"], rf_metrics["test_rps"]]
bars = ax.bar(["XGBoost v4", "Random Forest"], vals, color=[colors["XGBoost v4"], colors["Random Forest"]], width=0.5)
for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width()/2, v + 0.0005, f"{v:.5f}", ha="center", fontsize=11, fontweight="bold")
ax.set_title("RPS — Ranked Probability Score (↓)", fontsize=13, fontweight="bold")
ax.set_ylim(min(vals) - 0.005, max(vals) + 0.005)

# Brier Score
ax = axes[1, 1]
vals = [xgb_metrics["test_brier"], rf_metrics["test_brier"]]
bars = ax.bar(["XGBoost v4", "Random Forest"], vals, color=[colors["XGBoost v4"], colors["Random Forest"]], width=0.5)
for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.5f}", ha="center", fontsize=11, fontweight="bold")
ax.set_title("Brier Score (↓)", fontsize=13, fontweight="bold")
ax.set_ylim(min(vals) - 0.01, max(vals) + 0.01)

# ECE
ax = axes[1, 2]
ece_labels = ["Home", "Draw", "Away", "Mean"]
xgb_ece_vals = [xgb_metrics["ece_home"], xgb_metrics["ece_draw"], xgb_metrics["ece_away"], xgb_metrics["ece_mean"]]
rf_ece_vals = [rf_metrics["ece_home"], rf_metrics["ece_draw"], rf_metrics["ece_away"], rf_metrics["ece_mean"]]
x = np.arange(len(ece_labels))
w = 0.3
bars1 = ax.bar(x - w/2, xgb_ece_vals, w, label="XGBoost v4", color=colors["XGBoost v4"])
bars2 = ax.bar(x + w/2, rf_ece_vals, w, label="Random Forest", color=colors["Random Forest"])
for b, v in zip(bars1, xgb_ece_vals): ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.4f}", ha="center", fontsize=8, rotation=45)
for b, v in zip(bars2, rf_ece_vals): ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.4f}", ha="center", fontsize=8, rotation=45)
ax.set_xticks(x)
ax.set_xticklabels(ece_labels)
ax.set_title("ECE — Expected Calibration Error (↓)", fontsize=13, fontweight="bold")
ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{RES_DIR}/xgboost_vs_rf_metrics.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/xgboost_vs_rf_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Metrics comparison chart saved.")

# --- 4b: Reliability diagrams (calibration) ---
xgb_cal = calibration_data(y_test, xgb_proba)
rf_cal = calibration_data(y_test, rf_proba)

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle("Diagrammes de Fiabilité — XGBoost v4 vs Random Forest", fontsize=16, fontweight="bold")

for idx, cls in enumerate(["Home", "Draw", "Away"]):
    ax = axes[idx]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Calibration parfaite")

    xc = xgb_cal[cls]
    rc = rf_cal[cls]

    ax.plot(xc["centers"], xc["actuals"], "o-", color=colors["XGBoost v4"],
            label=f"XGBoost v4 (ECE={xgb_ece[cls]:.4f})", markersize=8, linewidth=2)
    ax.plot(rc["centers"], rc["actuals"], "s-", color=colors["Random Forest"],
            label=f"Random Forest (ECE={rf_ece[cls]:.4f})", markersize=8, linewidth=2)

    ax.set_xlabel("Probabilité prédite", fontsize=12)
    ax.set_ylabel("Fréquence observée", fontsize=12)
    ax.set_title(f"Victoire {cls}" if cls != "Draw" else "Match Nul", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RES_DIR}/xgboost_vs_rf_calibration.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/xgboost_vs_rf_calibration.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Calibration chart saved.")

# --- 4c: Feature importance comparison ---
xgb_imp = gs_xgb.best_estimator_.feature_importances_
rf_imp = gs_rf.best_estimator_.feature_importances_

fi_df = pd.DataFrame({
    "feature": ADVANCED_FEATURES,
    "xgb_importance": xgb_imp,
    "rf_importance": rf_imp,
})
fi_df["avg_importance"] = (fi_df["xgb_importance"] + fi_df["rf_importance"]) / 2
fi_df = fi_df.sort_values("avg_importance", ascending=True)
fi_df.to_csv(f"{RES_DIR}/feature_importance_xgb_vs_rf.csv", index=False)

# Plot top 20 features
top_n = min(20, len(fi_df))
fi_top = fi_df.tail(top_n)

fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.45)))
y_pos = np.arange(top_n)
w = 0.35

bars1 = ax.barh(y_pos - w/2, fi_top["xgb_importance"], w,
                label="XGBoost v4", color=colors["XGBoost v4"], alpha=0.85)
bars2 = ax.barh(y_pos + w/2, fi_top["rf_importance"], w,
                label="Random Forest", color=colors["Random Forest"], alpha=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels(fi_top["feature"], fontsize=10)
ax.set_xlabel("Feature Importance", fontsize=12)
ax.set_title("Top 20 Features — XGBoost v4 vs Random Forest", fontsize=15, fontweight="bold")
ax.legend(fontsize=12, loc="lower right")

# Color feature labels: orange for advanced, blue for baseline
for label in ax.get_yticklabels():
    fname = label.get_text()
    if fname in ADVANCED_ROLLING:
        label.set_color("#FF6F00")
    elif fname in ["home_played_europe", "away_played_europe"]:
        label.set_color("#E53935")
    else:
        label.set_color("#1565C0")

plt.tight_layout()
plt.savefig(f"{RES_DIR}/feature_importance_xgb_vs_rf.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/feature_importance_xgb_vs_rf.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Feature importance chart saved.")

# --- 4d: Confusion matrices side by side ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Matrices de Confusion — XGBoost v4 vs Random Forest", fontsize=16, fontweight="bold")

for ax, pred, name in [(axes[0], xgb_pred, "XGBoost v4"), (axes[1], rf_pred, "Random Forest")]:
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues" if "XGB" in name else "Greens",
                xticklabels=["Home", "Draw", "Away"],
                yticklabels=["Home", "Draw", "Away"], ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    acc = accuracy_score(y_test, pred)
    ax.set_title(f"{name} (Accuracy: {acc:.4f})", fontsize=13)

plt.tight_layout()
plt.savefig(f"{RES_DIR}/confusion_matrix_xgb_vs_rf.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/confusion_matrix_xgb_vs_rf.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Confusion matrix chart saved.")

# --- 4e: Probability distribution comparison ---
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle("Distribution des Probabilités Prédites — XGBoost v4 vs Random Forest",
             fontsize=16, fontweight="bold")

for idx, cls in enumerate(["Home", "Draw", "Away"]):
    ax = axes[idx]
    ax.hist(xgb_proba[:, idx], bins=30, alpha=0.6, color=colors["XGBoost v4"],
            label="XGBoost v4", density=True)
    ax.hist(rf_proba[:, idx], bins=30, alpha=0.6, color=colors["Random Forest"],
            label="Random Forest", density=True)
    ax.set_xlabel("Probabilité prédite")
    ax.set_ylabel("Densité")
    title = f"Victoire {cls}" if cls != "Draw" else "Match Nul"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()

plt.tight_layout()
plt.savefig(f"{RES_DIR}/prob_distribution_xgb_vs_rf.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/prob_distribution_xgb_vs_rf.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Probability distribution chart saved.")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: Summary report
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Summary")
print("=" * 70)

xgb_wins = sum(1 for w in better if w == "XGBoost")
rf_wins = sum(1 for w in better if w == "Random Forest")
ties = sum(1 for w in better if w == "Tie")

print(f"\n  XGBoost v4 wins: {xgb_wins} metrics")
print(f"  Random Forest wins: {rf_wins} metrics")
print(f"  Ties: {ties}")
print(f"\n  Overall winner: {'XGBoost v4' if xgb_wins > rf_wins else 'Random Forest' if rf_wins > xgb_wins else 'Tie'}")

# Save all results
all_results = {
    "xgboost_v4": {
        "best_params": xgb_best_params,
        "metrics": xgb_metrics,
    },
    "random_forest": {
        "best_params": {k: str(v) for k, v in rf_best_params.items()},
        "metrics": rf_metrics,
    },
    "comparison": {
        "xgb_wins": xgb_wins,
        "rf_wins": rf_wins,
        "ties": ties,
    },
}

with open(f"{RES_DIR}/xgboost_vs_rf_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print("\nDONE.")
