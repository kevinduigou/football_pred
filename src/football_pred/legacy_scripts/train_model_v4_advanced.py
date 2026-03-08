"""
Train and validate XGBoost model v4 with advanced match statistics rolling averages.

Compares three models:
1. Baseline v2 (15 features)
2. Europe v3 (17 features)
3. Advanced v4 (17 + 16 rolling avg features = 33 features, xG excluded - not available)

Uses the enriched dataset with rolling averages.
Handles NaN values with XGBoost's native missing value support.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    log_loss, classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "/home/ubuntu/football_pred/results"
DATASET_PATH = "/home/ubuntu/football_pred/dataset/football_matches_v4.csv"
MODEL_DIR = "/home/ubuntu/football_pred/models"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv(DATASET_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"Dataset: {df.shape[0]} matches, {df.shape[1]} columns")
print(f"Seasons: {sorted(df['season'].unique())}")

# --------------------------------------------------
# 2. Define feature sets
# --------------------------------------------------
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

# Advanced features: Europe + rolling averages (exclude xG which has 0 coverage)
ADVANCED_ROLLING = [
    "home_shots_on_goal_avg5", "away_shots_on_goal_avg5",
    "home_total_shots_avg5", "away_total_shots_avg5",
    "home_shots_insidebox_avg5", "away_shots_insidebox_avg5",
    "home_ball_possession_avg5", "away_ball_possession_avg5",
    "home_total_passes_avg5", "away_total_passes_avg5",
    "home_passes_pct_avg5", "away_passes_pct_avg5",
    "home_corner_kicks_avg5", "away_corner_kicks_avg5",
    "home_fouls_avg5", "away_fouls_avg5",
]

ADVANCED_FEATURES = EUROPE_FEATURES + ADVANCED_ROLLING

y = df["result"].map({"H": 0, "D": 1, "A": 2})

print(f"\nTarget distribution:")
print(f"  Home Win (0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"  Draw (1):     {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
print(f"  Away Win (2): {(y==2).sum()} ({(y==2).mean()*100:.1f}%)")

# Coverage of advanced features
adv_coverage = df[ADVANCED_ROLLING[0]].notna().sum()
print(f"\nAdvanced stats coverage: {adv_coverage}/{len(df)} ({adv_coverage/len(df)*100:.1f}%)")

# --------------------------------------------------
# 3. Chronological train/test split (same for all models)
# --------------------------------------------------
n = len(df)
split_idx = int(n * 0.8)

y_train_full = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"\nTrain: {split_idx}, Test: {n - split_idx}")
print(f"Test set seasons: {sorted(df.iloc[split_idx:]['season'].unique())}")

# Check advanced stats coverage in test set
test_adv = df.iloc[split_idx:][ADVANCED_ROLLING[0]].notna().sum()
print(f"Test set advanced stats coverage: {test_adv}/{n - split_idx} ({test_adv/(n-split_idx)*100:.1f}%)")

# --------------------------------------------------
# 4. Train and evaluate all models
# --------------------------------------------------
results_summary = {}

model_configs = [
    ("baseline_v2", BASELINE_FEATURES),
    ("europe_v3", EUROPE_FEATURES),
    ("advanced_v4", ADVANCED_FEATURES),
]

for model_name, feature_cols in model_configs:
    print(f"\n{'='*65}")
    print(f"TRAINING MODEL: {model_name} ({len(feature_cols)} features)")
    print(f"{'='*65}")

    X = df[feature_cols]
    X_train_full = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    # Further split for early stopping
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    # Hyperparameter tuning
    print(f"\n--- Hyperparameter Tuning (3-fold CV) ---")
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [4, 5, 6],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.7, 0.8],
        "min_child_weight": [3, 5],
    }

    xgb_base = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",  # Handles NaN natively
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        xgb_base, param_grid, cv=cv,
        scoring="neg_log_loss", n_jobs=-1, verbose=0, refit=True,
    )
    grid_search.fit(X_train_full, y_train_full)

    best_params = grid_search.best_params_
    cv_logloss = -grid_search.best_score_
    print(f"Best parameters: {best_params}")
    print(f"CV Log Loss: {cv_logloss:.4f}")

    # Train final model with early stopping evaluation
    xgb_final = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        **best_params,
    )
    xgb_final.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=False,
    )

    # Evaluate on test set
    proba_test = xgb_final.predict_proba(X_test)
    pred_test = xgb_final.predict(X_test)
    test_logloss = log_loss(y_test, proba_test)
    test_acc = accuracy_score(y_test, pred_test)
    test_f1_weighted = f1_score(y_test, pred_test, average="weighted")
    test_f1_macro = f1_score(y_test, pred_test, average="macro")

    print(f"\n--- Test Set ---")
    print(f"Log Loss: {test_logloss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Weighted: {test_f1_weighted:.4f}")
    print(f"F1 Macro: {test_f1_macro:.4f}")
    report = classification_report(y_test, pred_test, target_names=["Home", "Draw", "Away"], output_dict=True)
    print(classification_report(y_test, pred_test, target_names=["Home", "Draw", "Away"]))

    results_summary[model_name] = {
        "features": len(feature_cols),
        "best_params": best_params,
        "cv_logloss": round(cv_logloss, 4),
        "test_logloss": round(test_logloss, 4),
        "test_accuracy": round(test_acc, 4),
        "test_f1_weighted": round(test_f1_weighted, 4),
        "test_f1_macro": round(test_f1_macro, 4),
        "classification_report": report,
    }

    # --------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------
    cm = confusion_matrix(y_test, pred_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Home", "Draw", "Away"],
                yticklabels=["Home", "Draw", "Away"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_{model_name}.png", dpi=150)
    plt.close()

    # --------------------------------------------------
    # Save feature importance
    # --------------------------------------------------
    importance = xgb_final.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(8, len(feature_cols) * 0.35)))
    colors = []
    for f_name in feat_imp["feature"]:
        if f_name in ADVANCED_ROLLING:
            colors.append("#FF6F00")  # Orange for new advanced features
        elif f_name in ["home_played_europe", "away_played_europe"]:
            colors.append("#E53935")  # Red for Europe features
        else:
            colors.append("#1565C0")  # Blue for baseline
    ax.barh(feat_imp["feature"], feat_imp["importance"], color=colors)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(f"Feature Importance - {model_name}")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1565C0", label="Baseline (v2)"),
        Patch(facecolor="#E53935", label="Europe (v3)"),
        Patch(facecolor="#FF6F00", label="Advanced Stats (v4)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance_{model_name}.png", dpi=150)
    plt.close()

    # --------------------------------------------------
    # Save training curves
    # --------------------------------------------------
    evals = xgb_final.evals_result()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(evals["validation_0"]["mlogloss"], label="Train", alpha=0.8)
    ax.plot(evals["validation_1"]["mlogloss"], label="Validation", alpha=0.8)
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("Multi-class Log Loss")
    ax.set_title(f"Training Curves - {model_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_curves_{model_name}.png", dpi=150)
    plt.close()

    # --------------------------------------------------
    # Save test predictions
    # --------------------------------------------------
    test_results = df.iloc[split_idx:][["date", "league_name", "home_team", "away_team", "result"]].copy()
    test_results = test_results.reset_index(drop=True)
    test_results["P_Home"] = proba_test[:, 0].round(4)
    test_results["P_Draw"] = proba_test[:, 1].round(4)
    test_results["P_Away"] = proba_test[:, 2].round(4)
    test_results["predicted"] = pd.Series(pred_test).map({0: "H", 1: "D", 2: "A"})
    test_results["correct"] = test_results["result"] == test_results["predicted"]
    test_results.to_csv(f"{OUTPUT_DIR}/test_predictions_{model_name}.csv", index=False)

    # --------------------------------------------------
    # Save production model (only for advanced_v4)
    # --------------------------------------------------
    if model_name == "advanced_v4":
        print(f"\n--- Retraining {model_name} on Full Dataset for Production ---")
        xgb_prod = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            tree_method="hist",
            **best_params,
        )
        xgb_prod.fit(X, y, verbose=False)
        xgb_prod.save_model(f"{MODEL_DIR}/xgb_football_model_v4_advanced.json")
        with open(f"{MODEL_DIR}/xgb_football_model_v4_advanced.pkl", "wb") as f:
            pickle.dump(xgb_prod, f)
        print(f"Production model saved to {MODEL_DIR}/")

# --------------------------------------------------
# 5. Comparison summary
# --------------------------------------------------
print("\n" + "=" * 80)
print("COMPARISON SUMMARY: v2 vs v3 vs v4")
print("=" * 80)

b = results_summary["baseline_v2"]
e = results_summary["europe_v3"]
a = results_summary["advanced_v4"]

comparison_data = {
    "Metric": [
        "Features",
        "CV Log Loss",
        "Test Log Loss",
        "Test Accuracy",
        "Test F1 (Weighted)",
        "Test F1 (Macro)",
        "Home Win F1",
        "Draw F1",
        "Away Win F1",
    ],
    "Baseline (v2)": [
        b["features"], b["cv_logloss"], b["test_logloss"],
        b["test_accuracy"], b["test_f1_weighted"], b["test_f1_macro"],
        round(b["classification_report"]["Home"]["f1-score"], 4),
        round(b["classification_report"]["Draw"]["f1-score"], 4),
        round(b["classification_report"]["Away"]["f1-score"], 4),
    ],
    "Europe (v3)": [
        e["features"], e["cv_logloss"], e["test_logloss"],
        e["test_accuracy"], e["test_f1_weighted"], e["test_f1_macro"],
        round(e["classification_report"]["Home"]["f1-score"], 4),
        round(e["classification_report"]["Draw"]["f1-score"], 4),
        round(e["classification_report"]["Away"]["f1-score"], 4),
    ],
    "Advanced (v4)": [
        a["features"], a["cv_logloss"], a["test_logloss"],
        a["test_accuracy"], a["test_f1_weighted"], a["test_f1_macro"],
        round(a["classification_report"]["Home"]["f1-score"], 4),
        round(a["classification_report"]["Draw"]["f1-score"], 4),
        round(a["classification_report"]["Away"]["f1-score"], 4),
    ],
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))
comparison_df.to_csv(f"{OUTPUT_DIR}/comparison_v2_v3_v4.csv", index=False)

# Save full results as JSON
with open(f"{OUTPUT_DIR}/results_summary_v4.json", "w") as f:
    json.dump(results_summary, f, indent=2, default=str)

# --------------------------------------------------
# 6. Create comparison visualizations
# --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

models = ["Baseline (v2)", "Europe (v3)", "Advanced (v4)"]
colors = ["#42A5F5", "#E53935", "#FF6F00"]

# Accuracy & F1 comparison
metrics = ["test_accuracy", "test_f1_weighted", "test_f1_macro"]
labels = ["Accuracy", "F1 (Weighted)", "F1 (Macro)"]
vals = {
    "Baseline (v2)": [b[m] for m in metrics],
    "Europe (v3)": [e[m] for m in metrics],
    "Advanced (v4)": [a[m] for m in metrics],
}

x = np.arange(len(labels))
width = 0.25
for i, (model, color) in enumerate(zip(models, colors)):
    axes[0].bar(x + i * width, vals[model], width, label=model, color=color)
axes[0].set_ylabel("Score")
axes[0].set_title("Accuracy & F1 Scores")
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(labels)
axes[0].legend()
axes[0].set_ylim(0, max(max(v) for v in vals.values()) * 1.15)

# Add value labels
for i, (model, color) in enumerate(zip(models, colors)):
    for j, v in enumerate(vals[model]):
        axes[0].text(j + i * width, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

# Log Loss comparison
ll_vals = [b["test_logloss"], e["test_logloss"], a["test_logloss"]]
bars = axes[1].bar(models, ll_vals, color=colors)
axes[1].set_ylabel("Log Loss (lower is better)")
axes[1].set_title("Test Log Loss")
axes[1].set_ylim(min(ll_vals) * 0.95, max(ll_vals) * 1.05)
for bar, v in zip(bars, ll_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

# Per-class F1 comparison
classes = ["Home", "Draw", "Away"]
for i, (model, color) in enumerate(zip(models, colors)):
    r = results_summary[list(results_summary.keys())[i]]["classification_report"]
    f1_vals = [r[c]["f1-score"] for c in classes]
    axes[2].bar(np.arange(len(classes)) + i * width, f1_vals, width, label=model, color=color)
axes[2].set_ylabel("F1 Score")
axes[2].set_title("Per-class F1 Scores")
axes[2].set_xticks(np.arange(len(classes)) + width)
axes[2].set_xticklabels(classes)
axes[2].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/comparison_chart_v4.png", dpi=150)
plt.close()

print(f"\nAll results saved to {OUTPUT_DIR}/")
print("Done!")
