# XGBoost Football Match Prediction Model

## Overview

This project builds a complete **XGBoost multiclass classifier** to predict the probability of three outcomes for a football match: **Home Win (H)**, **Draw (D)**, and **Away Win (A)**. The model was trained on real match data collected from the top 5 European football leagues via the [API-Football](https://www.api-football.com/) service.

## Data Collection

Match data was scraped from the API-Football v3 endpoint using the provided API key. The following leagues and seasons were collected:

| League | Country | Seasons | Matches |
|--------|---------|---------|---------|
| Premier League | England | 2022–2025 | 1,140 |
| La Liga | Spain | 2022–2025 | 1,140 |
| Serie A | Italy | 2022–2025 | 1,141 |
| Bundesliga | Germany | 2022–2025 | 923 |
| Ligue 1 | France | 2022–2025 | 994 |
| **Total** | | | **5,338** |

After a warm-up period of 300 matches (required to stabilize ELO ratings and rolling statistics), the final dataset contains **5,038 matches**.

## Feature Engineering

Fifteen features were engineered from the raw match data. These features capture team strength, recent form, scoring patterns, rest advantage, and head-to-head history.

| Feature | Description |
|---------|-------------|
| `home_elo` | Pre-match ELO rating of the home team (K=20, initial=1500, home advantage=50) |
| `away_elo` | Pre-match ELO rating of the away team |
| `elo_diff` | Difference: home_elo − away_elo |
| `home_form_5` | Average points per game over the last 5 matches (home team) |
| `away_form_5` | Average points per game over the last 5 matches (away team) |
| `home_goals_for_avg` | Average goals scored per game over the last 5 matches (home team) |
| `away_goals_for_avg` | Average goals scored per game over the last 5 matches (away team) |
| `home_goals_against_avg` | Average goals conceded per game over the last 5 matches (home team) |
| `away_goals_against_avg` | Average goals conceded per game over the last 5 matches (away team) |
| `home_rest_days` | Days since the home team's last match (capped at 30) |
| `away_rest_days` | Days since the away team's last match (capped at 30) |
| `home_h2h_win_rate` | Home team's win rate in the last 5 head-to-head meetings |
| `away_h2h_win_rate` | Away team's win rate in the last 5 head-to-head meetings |
| `home_gd_form` | Average goal difference per game over the last 5 matches (home team) |
| `away_gd_form` | Average goal difference per game over the last 5 matches (away team) |

## Model Architecture

The model uses **XGBoost** with the `multi:softprob` objective to output calibrated probabilities for each of the three classes. Hyperparameters were tuned via **5-fold stratified cross-validation** with **GridSearchCV**, optimizing for multi-class log loss.

### Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 200 |
| `max_depth` | 4 |
| `learning_rate` | 0.03 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 3 |

## Evaluation Results

The model was evaluated using a **chronological hold-out** strategy: the last 20% of matches (1,008 matches) serve as the test set, ensuring no future data leakage.

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Log Loss** | 1.0177 |
| **Test Accuracy** | 51.2% |
| **CV Log Loss** | 1.0096 |
| Baseline Accuracy (always Home) | 40.3% |

### Classification Report (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Home Win | 0.51 | 0.77 | 0.61 | 406 |
| Draw | 0.26 | 0.03 | 0.06 | 245 |
| Away Win | 0.54 | 0.55 | 0.54 | 357 |
| **Weighted Avg** | **0.46** | **0.51** | **0.45** | **1,008** |

The model performs well at predicting Home and Away wins but struggles with Draws, which is a well-known challenge in football prediction. However, the **probability outputs** are more informative than the hard class predictions, as the model assigns meaningful draw probabilities even when the predicted class is Home or Away.

## Feature Importance

The **ELO difference** (`elo_diff`) is by far the most important feature, contributing roughly 17.5% of the model's gain. The individual ELO ratings (`home_elo`, `away_elo`) follow at around 7–8%. All other features (form, goals, rest days, head-to-head) contribute roughly equally at 5–6% each, indicating a well-balanced feature set.

![Feature Importance](https://private-us-east-1.manuscdn.com/sessionFile/nav6skRLYJpBPRv6mezmfw/sandbox/nVVkh3a278DPif1HCxPX2J-images_1772835738422_na1fn_L2hvbWUvdWJ1bnR1L2ZlYXR1cmVfaW1wb3J0YW5jZQ.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvbmF2NnNrUkxZSnBCUFJ2Nm1lem1mdy9zYW5kYm94L25WVmtoM2EyNzhEUGlmMUhDeFBYMkotaW1hZ2VzXzE3NzI4MzU3Mzg0MjJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyWmxZWFIxY21WZmFXMXdiM0owWVc1alpRLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=focnyzZ8eb00Tyac8AU0e63a-Nqw6Rz9eMhJ6tzyfljNWtkAmKOPGrCGbNzQPgIhZm7zvCHzv914VFJ8SjsimZGLJc7P1IaU~Wv7wufu~gnF0fghuKzXbR9KLU7w3k7sKFN9rqpynUyCvBHLFGjZ74ERydg6ZHu0xFKr-rgszxo1OP2XutUyIzoKUrmL2ijHqnogkDlvYkShEg1eP2-VhFe9i-2kYMpNL8~L5ahKBBLWQ9LqKNp7AqDAmht8Nt8uPpl0CrWHHSECwdIw1HpbaQU0946d1e6aJ8Uy2fuC6Hv6ZgnOsfCEW~mHU6~njuuo-mbp21sHcnhSbu86~YoF~A__)

## Example Prediction

For a hypothetical future match with the following characteristics:

| Feature | Value |
|---------|-------|
| Home ELO | 1620 |
| Away ELO | 1580 |
| ELO Diff | +40 |
| Home Form | 2.0 pts/game |
| Away Form | 1.2 pts/game |

The model predicts:

| Outcome | Probability | Implied Odds |
|---------|-------------|--------------|
| **Home Win** | **57.9%** | **1.73** |
| Draw | 24.4% | 4.09 |
| Away Win | 17.7% | 5.65 |

## Files Delivered

| File | Description |
|------|-------------|
| `football_matches.csv` | Complete dataset with 5,038 matches and 15 engineered features |
| `train_model.py` | Full training pipeline: data loading, tuning, training, evaluation, visualization |
| `collect_data.py` | API data collection script |
| `build_features.py` | Feature engineering script |
| `xgb_football_model.json` | Saved XGBoost model (JSON format) |
| `xgb_football_model.pkl` | Saved XGBoost model (Pickle format) |
| `test_predictions.csv` | Predictions on the 1,008 test matches |
| `feature_importance.png` | Feature importance visualization |
| `confusion_matrix.png` | Confusion matrix visualization |
| `training_curves.png` | Training/validation loss curves |
| `calibration.png` | Probability calibration plots |

## How to Use

```python
import pandas as pd
import pickle

# Load model
with open("xgb_football_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare features for a new match
match = pd.DataFrame([{
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

# Predict probabilities
proba = model.predict_proba(match)[0]
print(f"P(Home): {proba[0]:.3f}")
print(f"P(Draw): {proba[1]:.3f}")
print(f"P(Away): {proba[2]:.3f}")
```
