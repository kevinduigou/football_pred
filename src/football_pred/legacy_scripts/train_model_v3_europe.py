"""
Train and validate XGBoost model v3 with European competition features.

This script:
1. Trains the BASELINE model (v2, 15 features) for reference metrics
2. Trains the NEW model (v3, 17 features including home_played_europe, away_played_europe)
3. Compares both models on the same chronological test set
4. Saves all results, plots, and the updated model
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
DATASET_PATH = "/home/ubuntu/football_pred/dataset/football_matches.csv"
MODEL_DIR = "/home/ubuntu/football_pred/models"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv(DATASET_PATH)
print(f"Dataset: {df.shape[0]} matches, {df.shape[1]} columns")
print(f"Seasons: {sorted(df['season'].unique())}")
print(f"Columns: {list(df.columns)}")

# Verify new columns exist
assert "home_played_europe" in df.columns, "home_played_europe column missing!"
assert "away_played_europe" in df.columns, "away_played_europe column missing!"

print(f"\nhome_played_europe distribution: {df['home_played_europe'].value_counts().to_dict()}")
print(f"away_played_europe distribution: {df['away_played_europe'].value_counts().to_dict()}")

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

y = df["result"].map({"H": 0, "D": 1, "A": 2})

print(f"\nTarget distribution:")
print(f"  Home Win (0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"  Draw (1):     {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
print(f"  Away Win (2): {(y==2).sum()} ({(y==2).mean()*100:.1f}%)")

# --------------------------------------------------
# 3. Chronological train/test split (same for both models)
# --------------------------------------------------
n = len(df)
split_idx = int(n * 0.8)

y_train_full = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"\nTrain: {split_idx}, Test: {n - split_idx}")
print(f"Test set covers seasons: {sorted(df.iloc[split_idx:]['season'].unique())}")

# --------------------------------------------------
# 4. Train and evaluate both models
# --------------------------------------------------
results_summary = {}

for model_name, feature_cols in [("baseline_v2", BASELINE_FEATURES), ("europe_v3", EUROPE_FEATURES)]:
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
        "max_depth": [4, 5],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "min_child_weight": [3, 5],
    }

    xgb_base = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
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
        **best_params,
    )
    xgb_final.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=False,
    )

    # Evaluate on validation set
    proba_valid = xgb_final.predict_proba(X_valid)
    pred_valid = xgb_final.predict(X_valid)
    valid_logloss = log_loss(y_valid, proba_valid)
    valid_acc = accuracy_score(y_valid, pred_valid)

    print(f"\n--- Validation Set ---")
    print(f"Log Loss: {valid_logloss:.4f}")
    print(f"Accuracy: {valid_acc:.4f}")
    print(classification_report(y_valid, pred_valid, target_names=["Home", "Draw", "Away"]))

    # Evaluate on test set
    proba_test = xgb_final.predict_proba(X_test)
    pred_test = xgb_final.predict(X_test)
    test_logloss = log_loss(y_test, proba_test)
    test_acc = accuracy_score(y_test, pred_test)
    test_f1_weighted = f1_score(y_test, pred_test, average="weighted")
    test_f1_macro = f1_score(y_test, pred_test, average="macro")

    print(f"\n--- Test Set (Chronological Hold-out) ---")
    print(f"Log Loss: {test_logloss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Weighted: {test_f1_weighted:.4f}")
    print(f"F1 Macro: {test_f1_macro:.4f}")
    report = classification_report(y_test, pred_test, target_names=["Home", "Draw", "Away"], output_dict=True)
    print(classification_report(y_test, pred_test, target_names=["Home", "Draw", "Away"]))

    # Store results
    results_summary[model_name] = {
        "features": len(feature_cols),
        "best_params": best_params,
        "cv_logloss": round(cv_logloss, 4),
        "valid_logloss": round(valid_logloss, 4),
        "valid_accuracy": round(valid_acc, 4),
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
    ax.set_title(f"Confusion Matrix — {model_name}")
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

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = []
    for f_name in feat_imp["feature"]:
        if f_name in ["home_played_europe", "away_played_europe"]:
            colors.append("#E53935")  # Red for new features
        elif feat_imp.loc[feat_imp["feature"] == f_name, "importance"].values[0] > 0.08:
            colors.append("#1565C0")
        elif feat_imp.loc[feat_imp["feature"] == f_name, "importance"].values[0] > 0.06:
            colors.append("#42A5F5")
        else:
            colors.append("#90CAF9")
    ax.barh(feat_imp["feature"], feat_imp["importance"], color=colors)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(f"Feature Importance — {model_name}")
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
    ax.set_title(f"Training Curves — {model_name}")
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
    # Retrain on ALL data for production (only for europe_v3)
    # --------------------------------------------------
    if model_name == "europe_v3":
        print(f"\n--- Retraining {model_name} on Full Dataset for Production ---")
        xgb_prod = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            **best_params,
        )
        xgb_prod.fit(X, y, verbose=False)

        xgb_prod.save_model(f"{MODEL_DIR}/xgb_football_model_v3_europe.json")
        with open(f"{MODEL_DIR}/xgb_football_model_v3_europe.pkl", "wb") as f:
            pickle.dump(xgb_prod, f)
        print(f"Production model saved to {MODEL_DIR}/")

# --------------------------------------------------
# 5. Comparison summary
# --------------------------------------------------
print("\n" + "=" * 80)
print("COMPARISON SUMMARY: BASELINE (v2) vs EUROPE (v3)")
print("=" * 80)

b = results_summary["baseline_v2"]
e = results_summary["europe_v3"]

comparison_data = {
    "Metric": [
        "Features",
        "CV Log Loss",
        "Validation Log Loss",
        "Validation Accuracy",
        "Test Log Loss",
        "Test Accuracy",
        "Test F1 (Weighted)",
        "Test F1 (Macro)",
        "Home Win Precision",
        "Home Win Recall",
        "Home Win F1",
        "Draw Precision",
        "Draw Recall",
        "Draw F1",
        "Away Win Precision",
        "Away Win Recall",
        "Away Win F1",
    ],
    "Baseline (v2)": [
        b["features"],
        b["cv_logloss"],
        b["valid_logloss"],
        b["valid_accuracy"],
        b["test_logloss"],
        b["test_accuracy"],
        b["test_f1_weighted"],
        b["test_f1_macro"],
        round(b["classification_report"]["Home"]["precision"], 4),
        round(b["classification_report"]["Home"]["recall"], 4),
        round(b["classification_report"]["Home"]["f1-score"], 4),
        round(b["classification_report"]["Draw"]["precision"], 4),
        round(b["classification_report"]["Draw"]["recall"], 4),
        round(b["classification_report"]["Draw"]["f1-score"], 4),
        round(b["classification_report"]["Away"]["precision"], 4),
        round(b["classification_report"]["Away"]["recall"], 4),
        round(b["classification_report"]["Away"]["f1-score"], 4),
    ],
    "Europe (v3)": [
        e["features"],
        e["cv_logloss"],
        e["valid_logloss"],
        e["valid_accuracy"],
        e["test_logloss"],
        e["test_accuracy"],
        e["test_f1_weighted"],
        e["test_f1_macro"],
        round(e["classification_report"]["Home"]["precision"], 4),
        round(e["classification_report"]["Home"]["recall"], 4),
        round(e["classification_report"]["Home"]["f1-score"], 4),
        round(e["classification_report"]["Draw"]["precision"], 4),
        round(e["classification_report"]["Draw"]["recall"], 4),
        round(e["classification_report"]["Draw"]["f1-score"], 4),
        round(e["classification_report"]["Away"]["precision"], 4),
        round(e["classification_report"]["Away"]["recall"], 4),
        round(e["classification_report"]["Away"]["f1-score"], 4),
    ],
}

comparison_df = pd.DataFrame(comparison_data)

# Compute delta
deltas = []
for i, row in comparison_df.iterrows():
    bv = row["Baseline (v2)"]
    ev = row["Europe (v3)"]
    if isinstance(bv, (int, float)) and isinstance(ev, (int, float)):
        diff = ev - bv
        # For log loss, lower is better
        metric = row["Metric"]
        if "Log Loss" in metric:
            direction = "better" if diff < 0 else ("worse" if diff > 0 else "same")
        else:
            direction = "better" if diff > 0 else ("worse" if diff < 0 else "same")
        deltas.append(f"{diff:+.4f} ({direction})")
    else:
        deltas.append("")

comparison_df["Delta"] = deltas

print(comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv(f"{OUTPUT_DIR}/comparison_baseline_vs_europe.csv", index=False)

# Save full results as JSON
with open(f"{OUTPUT_DIR}/results_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2, default=str)

# --------------------------------------------------
# 6. Create comparison visualization
# --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Accuracy comparison
metrics_acc = ["test_accuracy", "test_f1_weighted", "test_f1_macro"]
labels_acc = ["Accuracy", "F1 (Weighted)", "F1 (Macro)"]
baseline_vals = [b[m] for m in metrics_acc]
europe_vals = [e[m] for m in metrics_acc]

x = np.arange(len(labels_acc))
width = 0.35
axes[0].bar(x - width/2, baseline_vals, width, label="Baseline (v2)", color="#42A5F5")
axes[0].bar(x + width/2, europe_vals, width, label="Europe (v3)", color="#E53935")
axes[0].set_ylabel("Score")
axes[0].set_title("Accuracy & F1 Scores")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels_acc)
axes[0].legend()
axes[0].set_ylim(0, max(max(baseline_vals), max(europe_vals)) * 1.15)

# Log Loss comparison
metrics_ll = ["cv_logloss", "valid_logloss", "test_logloss"]
labels_ll = ["CV Log Loss", "Valid Log Loss", "Test Log Loss"]
baseline_ll = [b[m] for m in metrics_ll]
europe_ll = [e[m] for m in metrics_ll]

x2 = np.arange(len(labels_ll))
axes[1].bar(x2 - width/2, baseline_ll, width, label="Baseline (v2)", color="#42A5F5")
axes[1].bar(x2 + width/2, europe_ll, width, label="Europe (v3)", color="#E53935")
axes[1].set_ylabel("Log Loss (lower is better)")
axes[1].set_title("Log Loss Comparison")
axes[1].set_xticks(x2)
axes[1].set_xticklabels(labels_ll, rotation=15)
axes[1].legend()

# Per-class F1 comparison
classes = ["Home", "Draw", "Away"]
baseline_f1 = [b["classification_report"][c]["f1-score"] for c in classes]
europe_f1 = [e["classification_report"][c]["f1-score"] for c in classes]

x3 = np.arange(len(classes))
axes[2].bar(x3 - width/2, baseline_f1, width, label="Baseline (v2)", color="#42A5F5")
axes[2].bar(x3 + width/2, europe_f1, width, label="Europe (v3)", color="#E53935")
axes[2].set_ylabel("F1-Score")
axes[2].set_title("Per-Class F1-Score")
axes[2].set_xticks(x3)
axes[2].set_xticklabels(classes)
axes[2].legend()

plt.suptitle("Model Comparison: Baseline (v2) vs Europe (v3)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/comparison_chart.png", dpi=150)
plt.close()

print(f"\nAll results saved to {OUTPUT_DIR}/")
print("Done!")
