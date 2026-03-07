import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import log_loss, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv("/home/ubuntu/football_matches.csv")
print(f"Dataset: {df.shape[0]} matches, {df.shape[1]} columns")

feature_cols = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form_5",
    "away_form_5",
    "home_goals_for_avg",
    "away_goals_for_avg",
    "home_goals_against_avg",
    "away_goals_against_avg",
    "home_rest_days",
    "away_rest_days",
    "home_h2h_win_rate",
    "away_h2h_win_rate",
    "home_gd_form",
    "away_gd_form",
]

X = df[feature_cols]
y = df["result"].map({"H": 0, "D": 1, "A": 2})

print(f"\nFeatures: {len(feature_cols)}")
print(f"Target distribution:\n{y.value_counts().sort_index()}")
print(f"  0 (Home): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"  1 (Draw): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
print(f"  2 (Away): {(y==2).sum()} ({(y==2).mean()*100:.1f}%)")

# --------------------------------------------------
# 2. Train / validation / test split
# --------------------------------------------------
# Use chronological split: last 20% as test
n = len(df)
split_idx = int(n * 0.8)

X_train_full = X.iloc[:split_idx]
y_train_full = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

# Further split train into train/validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"\nTrain: {len(X_train)}, Validation: {len(X_valid)}, Test: {len(X_test)}")

# --------------------------------------------------
# 3. Hyperparameter tuning with GridSearchCV
# --------------------------------------------------
print("\n=== Hyperparameter Tuning ===")

param_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.08],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "min_child_weight": [3, 5],
}

xgb_base = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    use_label_encoder=False,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    xgb_base,
    param_grid,
    cv=cv,
    scoring="neg_log_loss",
    n_jobs=-1,
    verbose=1,
    refit=True,
)

grid_search.fit(X_train_full, y_train_full)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV log loss: {-grid_search.best_score_:.4f}")

# --------------------------------------------------
# 4. Train final model with best params
# --------------------------------------------------
print("\n=== Training Final Model ===")

best_params = grid_search.best_params_
xgb_final = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    use_label_encoder=False,
    **best_params,
)

xgb_final.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=False,
)

# --------------------------------------------------
# 5. Evaluate on validation and test sets
# --------------------------------------------------
print("\n=== Validation Set Evaluation ===")
proba_valid = xgb_final.predict_proba(X_valid)
pred_valid = xgb_final.predict(X_valid)
print(f"Log loss: {log_loss(y_valid, proba_valid):.4f}")
print(f"Accuracy: {accuracy_score(y_valid, pred_valid):.4f}")
print(classification_report(y_valid, pred_valid, target_names=["Home", "Draw", "Away"]))

print("\n=== Test Set Evaluation (Chronological Hold-out) ===")
proba_test = xgb_final.predict_proba(X_test)
pred_test = xgb_final.predict(X_test)
print(f"Log loss: {log_loss(y_test, proba_test):.4f}")
print(f"Accuracy: {accuracy_score(y_test, pred_test):.4f}")
print(classification_report(y_test, pred_test, target_names=["Home", "Draw", "Away"]))

# --------------------------------------------------
# 6. Retrain on ALL data with best params for production
# --------------------------------------------------
print("\n=== Retraining on Full Dataset ===")
xgb_prod = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    use_label_encoder=False,
    **best_params,
)
xgb_prod.fit(X, y, verbose=False)

# Save model
xgb_prod.save_model("/home/ubuntu/xgb_football_model.json")
with open("/home/ubuntu/xgb_football_model.pkl", "wb") as f:
    pickle.dump(xgb_prod, f)
print("Model saved: xgb_football_model.json / .pkl")

# --------------------------------------------------
# 7. Feature Importance
# --------------------------------------------------
importance = xgb_final.feature_importances_
feat_imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": importance
}).sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(feat_imp["feature"], feat_imp["importance"], color="#2196F3")
ax.set_xlabel("Feature Importance (Gain)")
ax.set_title("XGBoost Feature Importance for Football Match Prediction")
plt.tight_layout()
plt.savefig("/home/ubuntu/feature_importance.png", dpi=150)
plt.close()
print("Feature importance plot saved.")

# --------------------------------------------------
# 8. Confusion Matrix
# --------------------------------------------------
cm = confusion_matrix(y_test, pred_test)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Home", "Draw", "Away"],
            yticklabels=["Home", "Draw", "Away"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("/home/ubuntu/confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix plot saved.")

# --------------------------------------------------
# 9. Training curves
# --------------------------------------------------
results = xgb_final.evals_result()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(results["validation_0"]["mlogloss"], label="Train")
ax.plot(results["validation_1"]["mlogloss"], label="Validation")
ax.set_xlabel("Boosting Round")
ax.set_ylabel("Multi-class Log Loss")
ax.set_title("XGBoost Training Curves")
ax.legend()
plt.tight_layout()
plt.savefig("/home/ubuntu/training_curves.png", dpi=150)
plt.close()
print("Training curves plot saved.")

# --------------------------------------------------
# 10. Probability calibration check
# --------------------------------------------------
# Check if predicted probabilities are well-calibrated
pred_df = pd.DataFrame(proba_test, columns=["P_Home", "P_Draw", "P_Away"])
pred_df["actual"] = y_test.values
pred_df["predicted"] = pred_test

# Binned calibration for Home win
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (col, label) in enumerate(zip(["P_Home", "P_Draw", "P_Away"], ["Home", "Draw", "Away"])):
    actual_class = (pred_df["actual"] == i).astype(int)
    bins = pd.qcut(pred_df[col], q=10, duplicates="drop")
    cal = pred_df.groupby(bins).apply(
        lambda g: pd.Series({
            "mean_predicted": g[col].mean(),
            "mean_actual": (g["actual"] == i).mean(),
            "count": len(g)
        })
    )
    axes[i].scatter(cal["mean_predicted"], cal["mean_actual"], s=cal["count"]*2, alpha=0.7)
    axes[i].plot([0, 1], [0, 1], "r--", alpha=0.5)
    axes[i].set_xlabel(f"Predicted P({label})")
    axes[i].set_ylabel(f"Actual P({label})")
    axes[i].set_title(f"Calibration: {label}")
plt.tight_layout()
plt.savefig("/home/ubuntu/calibration.png", dpi=150)
plt.close()
print("Calibration plot saved.")

# --------------------------------------------------
# 11. Predict one future match (example)
# --------------------------------------------------
print("\n=== Example Prediction: Future Match ===")
next_match = pd.DataFrame([{
    "home_elo": 1620,
    "away_elo": 1580,
    "elo_diff": 40,
    "home_form_5": 2.0,
    "away_form_5": 1.2,
    "home_goals_for_avg": 1.8,
    "away_goals_for_avg": 1.3,
    "home_goals_against_avg": 0.9,
    "away_goals_against_avg": 1.1,
    "home_rest_days": 6,
    "away_rest_days": 4,
    "home_h2h_win_rate": 0.5,
    "away_h2h_win_rate": 0.3,
    "home_gd_form": 0.8,
    "away_gd_form": -0.2,
}])

next_proba = xgb_prod.predict_proba(next_match)[0]
print(f"  P(Home Win): {next_proba[0]:.4f} ({next_proba[0]*100:.1f}%)")
print(f"  P(Draw):     {next_proba[1]:.4f} ({next_proba[1]*100:.1f}%)")
print(f"  P(Away Win): {next_proba[2]:.4f} ({next_proba[2]*100:.1f}%)")

# Implied odds
print(f"\n  Implied odds:")
print(f"    Home: {1/next_proba[0]:.2f}")
print(f"    Draw: {1/next_proba[1]:.2f}")
print(f"    Away: {1/next_proba[2]:.2f}")

# --------------------------------------------------
# 12. Save prediction probabilities for test set
# --------------------------------------------------
test_results = df.iloc[split_idx:][["date", "league_name", "home_team", "away_team", "result"]].copy()
test_results = test_results.reset_index(drop=True)
test_results["P_Home"] = proba_test[:, 0].round(4)
test_results["P_Draw"] = proba_test[:, 1].round(4)
test_results["P_Away"] = proba_test[:, 2].round(4)
test_results["predicted"] = pd.Series(pred_test).map({0: "H", 1: "D", 2: "A"})
test_results["correct"] = test_results["result"] == test_results["predicted"]
test_results.to_csv("/home/ubuntu/test_predictions.csv", index=False)
print(f"\nTest predictions saved: {len(test_results)} matches")

# --------------------------------------------------
# 13. Summary statistics
# --------------------------------------------------
print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
print(f"Training data:     {len(X_train_full)} matches")
print(f"Test data:         {len(X_test)} matches (chronological)")
print(f"Features:          {len(feature_cols)}")
print(f"Best parameters:   {best_params}")
print(f"CV Log Loss:       {-grid_search.best_score_:.4f}")
print(f"Test Log Loss:     {log_loss(y_test, proba_test):.4f}")
print(f"Test Accuracy:     {accuracy_score(y_test, pred_test):.4f}")
print(f"Baseline (always Home): {(y_test==0).mean():.4f}")
print("="*60)
