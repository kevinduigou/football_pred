# XGBoost Football Match Prediction Model (v2 — Extended Dataset)

## Overview

This project builds a complete **XGBoost multiclass classifier** to predict the probability of three outcomes for a football match: **Home Win (H)**, **Draw (D)**, and **Away Win (A)**. The model was trained on real match data collected from the top 5 European football leagues via the [API-Football](https://www.api-football.com/) service, spanning **8 full seasons from 2017 to 2025**.

## Data Collection

Match data was scraped from the API-Football v3 endpoint. The following leagues and seasons were collected:

| League | Country | Seasons | Matches |
|--------|---------|---------|---------|
| Premier League | England | 2017–2025 | 3,040 |
| La Liga | Spain | 2017–2025 | 3,040 |
| Serie A | Italy | 2017–2025 | 3,041 |
| Bundesliga | Germany | 2017–2025 | 2,461 |
| Ligue 1 | France | 2017–2025 | 2,796 |
| **Total** | | | **14,378** |

After a warm-up period of 500 matches (required to stabilize ELO ratings and rolling statistics), the final dataset contains **13,878 matches**.

## Comparison: v1 (3 seasons) vs v2 (8 seasons)

| Metric | v1 (2022-2025) | v2 (2017-2025) | Improvement |
|--------|:--------------:|:--------------:|:-----------:|
| Training matches | 4,030 | 11,102 | +175% |
| Test matches | 1,008 | 2,776 | +175% |
| CV Log Loss | 1.0096 | **0.9942** | -1.5% |
| Test Log Loss | 1.0177 | **0.9932** | -2.4% |
| Test Accuracy | 51.2% | **52.6%** | +1.4pp |
| ELO warm-up depth | 300 matches | 500 matches | More stable |
| Teams with ELO history | 122 | 150 | +23% |

The extended dataset provides a meaningful improvement in both log loss and accuracy, with more stable ELO ratings thanks to the deeper historical data.

## Feature Engineering

Fifteen features were engineered from the raw match data:

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

## Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 200 |
| `max_depth` | 4 |
| `learning_rate` | 0.03 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 3 |

## Test Set Results (2,776 matches — seasons 2023-2025)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Home Win | 0.53 | 0.80 | 0.64 | 1,167 |
| Draw | 0.17 | 0.01 | 0.01 | 706 |
| Away Win | 0.53 | 0.58 | 0.55 | 903 |
| **Weighted Avg** | **0.44** | **0.53** | **0.45** | **2,776** |

## Feature Importance

With 8 seasons of data, the **ELO difference** (`elo_diff`) becomes even more dominant at ~26% importance (up from 17.5% with 3 seasons). This makes sense: with more historical data, the ELO ratings are better calibrated and more predictive.

![Feature Importance](https://private-us-east-1.manuscdn.com/sessionFile/nav6skRLYJpBPRv6mezmfw/sandbox/himNzAh9lD7FeRLpa3l8Ya-images_1772866856730_na1fn_L2hvbWUvdWJ1bnR1L2ZlYXR1cmVfaW1wb3J0YW5jZQ.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvbmF2NnNrUkxZSnBCUFJ2Nm1lem1mdy9zYW5kYm94L2hpbU56QWg5bEQ3RmVSTHBhM2w4WWEtaW1hZ2VzXzE3NzI4NjY4NTY3MzBfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyWmxZWFIxY21WZmFXMXdiM0owWVc1alpRLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=hbja81DGsyEZ1aAFF5xcWgmpLY6CovTROnYAyw~3ZWRCpHltUNKM94dKatCCbe5zGVHs2kKD-2~k9~9NN1XV4MVisIp1Ge70VBb788hWLzF4geu45VP1bfPgaO7dWnb2t1h2MBQUofiVwyaEwNTqy3g7-NTpDUhqBHRS3g~wAmvZPqlvYm~341vg3OLpNlaPahxMab5dxFvtQXQSrfHQzJPB4VbdMNzh6kQgbhjfPrIgp~pTnNpPjom6t0WLtwMC8GAI6DWI0lnhEzZReVrWVZtInQHDzsD~3VvOgA1TtKhdV6EkOzZMj3xtZJNcdMDyt6WYa1KqmqY4GROGVMI4Cw__)

## Files Delivered

| File | Description |
|------|-------------|
| `football_matches.csv` | Complete dataset: 13,878 matches, 8 seasons, 15 features |
| `train_model_v2.py` | Full training pipeline (v2) |
| `collect_data.py` | API collection: seasons 2022-2024 |
| `collect_data_extended.py` | API collection: seasons 2017-2021 |
| `build_features_extended.py` | Feature engineering (extended) |
| `xgb_football_model.json` | Saved model (JSON) |
| `xgb_football_model.pkl` | Saved model (Pickle) |
| `test_predictions.csv` | Predictions on 2,776 test matches |
| `elo_ratings_all_teams.csv` | ELO ratings for all 150 teams |
| `feature_importance.png` | Feature importance chart |
| `confusion_matrix.png` | Confusion matrix |
| `training_curves.png` | Training/validation loss curves |
| `calibration.png` | Probability calibration plots |
