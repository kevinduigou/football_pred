#!/usr/bin/env python3
"""
Update Football Dataset Script

This script fetches the latest match results from API-Football for the 5 major European leagues,
collects their advanced statistics, and updates the main dataset (football_matches_v4.csv)
by recalculating all features (ELO, form, rolling averages, etc.) consistently with the existing data.

Usage:
    export FOOTBALL_API_SPORTS="your_api_key"
    python3 update_dataset.py
"""

import os
import sys
import json
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/update_dataset.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
API_KEY = os.environ.get("FOOTBALL_API_SPORTS")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# Top 5 European leagues
LEAGUES = {
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
}
EUROPEAN_LEAGUE_IDS = [2, 3, 848]  # UCL, UEL, UECL

# File paths
DATA_DIR = "/home/ubuntu/football_pred/dataset"
RAW_FIXTURES_FILE = os.path.join(DATA_DIR, "raw_fixtures_extended.json")
EURO_CACHE_FILE = os.path.join(DATA_DIR, "european_fixtures_cache.json")
STATS_CACHE_FILE = os.path.join(DATA_DIR, "advanced_stats_cache.json")
FINAL_DATASET_FILE = os.path.join(DATA_DIR, "football_matches_v4.csv")
ELO_RATINGS_FILE = os.path.join(DATA_DIR, "elo_ratings_all_teams.csv")

# Feature calculation constants
K = 20  # ELO K-factor
INITIAL_ELO = 1500
HOME_ADVANTAGE = 50
FORM_WINDOW = 5
GOALS_WINDOW = 5
ROLLING_WINDOW = 5

# Advanced stats to collect
STAT_TYPES = [
    "Shots on Goal", "Shots off Goal", "Total Shots", "Blocked Shots",
    "Shots insidebox", "Shots outsidebox",
    "Fouls", "Corner Kicks", "Offsides",
    "Ball Possession", "Yellow Cards", "Red Cards",
    "Goalkeeper Saves", "Total passes", "Passes accurate", "Passes %",
    "expected_goals", "goals_prevented",
]

ROLLING_STATS = [
    "shots_on_goal", "total_shots", "shots_insidebox", "ball_possession",
    "total_passes", "passes_pct", "corner_kicks", "fouls", "expected_goals",
]

def check_api_key():
    """Check if API key is set and valid."""
    if not API_KEY:
        logger.error("FOOTBALL_API_SPORTS environment variable is not set.")
        sys.exit(1)
        
    try:
        r = requests.get(f"{BASE_URL}/status", headers=HEADERS, timeout=10)
        data = r.json()
        if data.get("errors") and len(data["errors"]) > 0:
            logger.error(f"API Error: {data['errors']}")
            sys.exit(1)
            
        req = data.get("response", {}).get("requests", {})
        cur = req.get("current", 0)
        lim = req.get("limit_day", 7500)
        logger.info(f"API Status: {cur}/{lim} requests used today.")
        
        if cur >= lim:
            logger.error("API daily limit reached.")
            sys.exit(1)
            
        return lim - cur
    except Exception as e:
        logger.error(f"Failed to check API status: {e}")
        sys.exit(1)

def load_json_cache(filepath, default_val=None):
    """Load a JSON cache file."""
    if default_val is None:
        default_val = [] if "fixtures" in filepath else {}
        
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
    return default_val

def save_json_cache(data, filepath):
    """Save data to a JSON cache file."""
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {e}")

def fetch_new_fixtures(existing_fixture_ids, current_season=2025):
    """Fetch finished fixtures that are not in the existing dataset."""
    new_fixtures = []
    
    for league_id, league_name in LEAGUES.items():
        logger.info(f"Checking for new matches in {league_name}...")
        try:
            r = requests.get(
                f"{BASE_URL}/fixtures", 
                headers=HEADERS, 
                params={"league": league_id, "season": current_season, "status": "FT"}
            )
            data = r.json()
            fixtures = data.get("response", [])
            
            league_new_count = 0
            for f in fixtures:
                fix_id = f["fixture"]["id"]
                if fix_id not in existing_fixture_ids:
                    match = {
                        "fixture_id": fix_id,
                        "date": f["fixture"]["date"],
                        "league_id": league_id,
                        "league_name": league_name,
                        "season": current_season,
                        "home_team_id": f["teams"]["home"]["id"],
                        "home_team": f["teams"]["home"]["name"],
                        "away_team_id": f["teams"]["away"]["id"],
                        "away_team": f["teams"]["away"]["name"],
                        "home_goals": f["goals"]["home"],
                        "away_goals": f["goals"]["away"],
                        "home_ht": f["score"]["halftime"]["home"],
                        "away_ht": f["score"]["halftime"]["away"],
                    }
                    
                    # Determine result
                    if match["home_goals"] is not None and match["away_goals"] is not None:
                        if match["home_goals"] > match["away_goals"]:
                            match["result"] = "H"
                        elif match["home_goals"] < match["away_goals"]:
                            match["result"] = "A"
                        else:
                            match["result"] = "D"
                        new_fixtures.append(match)
                        league_new_count += 1
            
            logger.info(f"  Found {league_new_count} new matches for {league_name}")
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Error fetching fixtures for {league_name}: {e}")
            
    return new_fixtures

def fetch_new_european_fixtures(existing_euro_ids, current_season=2025):
    """Fetch new European fixtures to update the Europe features."""
    new_euro_fixtures = []
    
    for league_id in EUROPEAN_LEAGUE_IDS:
        league_name = {2: "Champions League", 3: "Europa League", 848: "Conference League"}[league_id]
        logger.info(f"Checking for new matches in {league_name}...")
        
        try:
            r = requests.get(
                f"{BASE_URL}/fixtures",
                headers=HEADERS,
                params={"league": league_id, "season": current_season, "status": "FT-AET-PEN"}
            )
            data = r.json()
            fixtures = data.get("response", [])
            
            league_new_count = 0
            for f in fixtures:
                fix_id = f["fixture"]["id"]
                if fix_id not in existing_euro_ids:
                    new_euro_fixtures.append({
                        "fixture_id": fix_id,
                        "date": f["fixture"]["date"],
                        "league_id": league_id,
                        "league_name": league_name,
                        "season": current_season,
                        "home_team_id": f["teams"]["home"]["id"],
                        "home_team": f["teams"]["home"]["name"],
                        "away_team_id": f["teams"]["away"]["id"],
                        "away_team": f["teams"]["away"]["name"],
                    })
                    league_new_count += 1
                    
            logger.info(f"  Found {league_new_count} new matches for {league_name}")
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error fetching fixtures for {league_name}: {e}")
            
    return new_euro_fixtures

def parse_stat_value(value):
    """Parse a statistic value from the API."""
    if value is None:
        return None
    if isinstance(value, str) and "%" in value:
        return float(value.replace("%", ""))
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def fetch_advanced_stats(fixture_ids, stats_cache):
    """Fetch advanced statistics for a list of fixture IDs."""
    to_fetch = [fid for fid in fixture_ids if str(fid) not in stats_cache]
    
    if not to_fetch:
        logger.info("All advanced stats already in cache.")
        return stats_cache
        
    logger.info(f"Fetching advanced stats for {len(to_fetch)} matches...")
    
    for i, fid in enumerate(to_fetch):
        try:
            r = requests.get(
                f"{BASE_URL}/fixtures/statistics",
                headers=HEADERS,
                params={"fixture": fid},
                timeout=15,
            )
            data = r.json()
            
            if data.get("errors"):
                logger.warning(f"  Error for fixture {fid}: {data['errors']}")
                continue
                
            response = data.get("response", [])
            if not response:
                stats_cache[str(fid)] = {}
                continue
                
            result = {}
            for team_data in response:
                tid = str(team_data.get("team", {}).get("id"))
                tname = team_data.get("team", {}).get("name")
                stats = {"team_name": tname}
                
                for s in team_data.get("statistics", []):
                    stats[s["type"]] = parse_stat_value(s.get("value"))
                    
                # Extract xG if available
                for s in team_data.get("statistics", []):
                    if "expected" in s["type"].lower() and "goal" in s["type"].lower():
                        stats["expected_goals"] = parse_stat_value(s.get("value"))
                        
                result[tid] = stats
                
            stats_cache[str(fid)] = result
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Fetched {i + 1}/{len(to_fetch)} stats...")
                save_json_cache(stats_cache, STATS_CACHE_FILE)
                
            time.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            logger.error(f"  Exception fetching stats for {fid}: {e}")
            
    save_json_cache(stats_cache, STATS_CACHE_FILE)
    return stats_cache

def clean_col_name(stat_name):
    """Clean a statistic name for use as a column name."""
    return stat_name.lower().replace(" ", "_").replace("%", "pct")

def match_team_stats(cache_entry, dataset_home, dataset_away):
    """Match API team names to dataset team names."""
    team_ids = list(cache_entry.keys())
    if len(team_ids) < 2:
        return {}, {}

    t1_stats = cache_entry[team_ids[0]]
    t2_stats = cache_entry[team_ids[1]]
    t1_name = t1_stats.get("team_name", "")
    t2_name = t2_stats.get("team_name", "")

    # Try exact match first
    if t1_name == dataset_home or t2_name == dataset_away:
        return t1_stats, t2_stats
    if t2_name == dataset_home or t1_name == dataset_away:
        return t2_stats, t1_stats

    # Fuzzy: check if one name contains the other
    dh = dataset_home.lower()
    da = dataset_away.lower()
    t1l = t1_name.lower()
    t2l = t2_name.lower()

    if t1l in dh or dh in t1l:
        return t1_stats, t2_stats
    if t2l in dh or dh in t2l:
        return t2_stats, t1_stats

    # Default: API returns home first
    return t1_stats, t2_stats

def played_europe_in_last_7_days(team_id, match_date, team_europe_dates):
    """Check if a team played a European match in the 7 days before match_date."""
    if team_id not in team_europe_dates:
        return 0

    match_date_naive = pd.to_datetime(match_date)
    if match_date_naive.tzinfo is not None:
        match_date_naive = match_date_naive.tz_localize(None)

    window_start = match_date_naive - timedelta(days=7)

    for euro_date in team_europe_dates[team_id]:
        if window_start <= euro_date < match_date_naive:
            return 1

    return 0

def recalculate_all_features(raw_df, euro_fixtures, stats_cache):
    """Recalculate all features from scratch to ensure consistency."""
    logger.info("Recalculating all features...")
    
    # Sort chronologically
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # 1. ELO Ratings
    logger.info("  Computing ELO ratings...")
    elo_ratings = defaultdict(lambda: INITIAL_ELO)
    home_elos, away_elos, elo_diffs = [], [], []
    
    for idx, row in df.iterrows():
        ht, at = row["home_team_id"], row["away_team_id"]
        home_elo, away_elo = elo_ratings[ht], elo_ratings[at]
        
        home_elos.append(home_elo)
        away_elos.append(away_elo)
        elo_diffs.append(home_elo - away_elo)
        
        exp_home = 1 / (1 + 10 ** ((away_elo - (home_elo + HOME_ADVANTAGE)) / 400))
        exp_away = 1 - exp_home
        
        if row["result"] == "H":
            actual_home, actual_away = 1.0, 0.0
        elif row["result"] == "A":
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5
            
        elo_ratings[ht] += K * (actual_home - exp_home)
        elo_ratings[at] += K * (actual_away - exp_away)
        
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = elo_diffs
    
    # Save updated ELO ratings
    team_names = {}
    team_league = {}
    team_match_count = defaultdict(int)
    for _, row in df.iterrows():
        team_names[row["home_team_id"]] = row["home_team"]
        team_names[row["away_team_id"]] = row["away_team"]
        team_league[row["home_team_id"]] = row["league_name"]
        team_league[row["away_team_id"]] = row["league_name"]
        team_match_count[row["home_team_id"]] += 1
        team_match_count[row["away_team_id"]] += 1

    elo_rows = []
    for tid in team_names:
        elo_rows.append({
            "Team": team_names[tid],
            "League": team_league[tid],
            "Initial ELO": INITIAL_ELO,
            "Final ELO": round(elo_ratings[tid], 1),
            "ELO Change": round(elo_ratings[tid] - INITIAL_ELO, 1),
            "Matches": team_match_count[tid],
        })
    pd.DataFrame(elo_rows).sort_values("Final ELO", ascending=False).to_csv(ELO_RATINGS_FILE, index=False)
    
    # 2. Form (points in last 5 matches)
    logger.info("  Computing form features...")
    team_history = defaultdict(list)
    home_form_5, away_form_5 = [], []
    
    for idx, row in df.iterrows():
        ht, at = row["home_team_id"], row["away_team_id"]
        h_hist, a_hist = team_history[ht], team_history[at]
        
        h_form = np.mean(h_hist[-FORM_WINDOW:]) if len(h_hist) >= FORM_WINDOW else (np.mean(h_hist) if h_hist else 1.0)
        a_form = np.mean(a_hist[-FORM_WINDOW:]) if len(a_hist) >= FORM_WINDOW else (np.mean(a_hist) if a_hist else 1.0)
        
        home_form_5.append(round(h_form, 3))
        away_form_5.append(round(a_form, 3))
        
        if row["result"] == "H":
            team_history[ht].append(3); team_history[at].append(0)
        elif row["result"] == "A":
            team_history[ht].append(0); team_history[at].append(3)
        else:
            team_history[ht].append(1); team_history[at].append(1)
            
    df["home_form_5"] = home_form_5
    df["away_form_5"] = away_form_5
    
    # 3. Goals averages
    logger.info("  Computing goals averages...")
    team_gf, team_ga = defaultdict(list), defaultdict(list)
    home_gf_avg, away_gf_avg, home_ga_avg, away_ga_avg = [], [], [], []
    
    for idx, row in df.iterrows():
        ht, at = row["home_team_id"], row["away_team_id"]
        hg, ag = row["home_goals"], row["away_goals"]
        
        h_gf, h_ga = team_gf[ht], team_ga[ht]
        a_gf, a_ga = team_gf[at], team_ga[at]
        
        h_gf_avg = np.mean(h_gf[-GOALS_WINDOW:]) if len(h_gf) >= GOALS_WINDOW else (np.mean(h_gf) if h_gf else 1.3)
        a_gf_avg = np.mean(a_gf[-GOALS_WINDOW:]) if len(a_gf) >= GOALS_WINDOW else (np.mean(a_gf) if a_gf else 1.3)
        h_ga_avg = np.mean(h_ga[-GOALS_WINDOW:]) if len(h_ga) >= GOALS_WINDOW else (np.mean(h_ga) if h_ga else 1.1)
        a_ga_avg = np.mean(a_ga[-GOALS_WINDOW:]) if len(a_ga) >= GOALS_WINDOW else (np.mean(a_ga) if a_ga else 1.1)
        
        home_gf_avg.append(round(h_gf_avg, 3)); away_gf_avg.append(round(a_gf_avg, 3))
        home_ga_avg.append(round(h_ga_avg, 3)); away_ga_avg.append(round(a_ga_avg, 3))
        
        team_gf[ht].append(hg); team_ga[ht].append(ag)
        team_gf[at].append(ag); team_ga[at].append(hg)
        
    df["home_goals_for_avg"] = home_gf_avg
    df["away_goals_for_avg"] = away_gf_avg
    df["home_goals_against_avg"] = home_ga_avg
    df["away_goals_against_avg"] = away_ga_avg
    
    # 4. Rest days
    logger.info("  Computing rest days...")
    team_last_match = {}
    home_rest, away_rest = [], []
    
    for idx, row in df.iterrows():
        ht, at, match_date = row["home_team_id"], row["away_team_id"], row["date"]
        
        h_rest = min((match_date - team_last_match[ht]).days, 30) if ht in team_last_match else 7
        a_rest = min((match_date - team_last_match[at]).days, 30) if at in team_last_match else 7
        
        home_rest.append(h_rest); away_rest.append(a_rest)
        team_last_match[ht] = match_date; team_last_match[at] = match_date
        
    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest
    
    # 5. H2H and GD form
    logger.info("  Computing H2H and GD form...")
    h2h_history = defaultdict(list)
    team_gd_history = defaultdict(list)
    home_h2h_wins, away_h2h_wins = [], []
    home_gd_form, away_gd_form = [], []
    
    for idx, row in df.iterrows():
        ht, at = row["home_team_id"], row["away_team_id"]
        key = (min(ht, at), max(ht, at))
        
        # H2H
        hist = h2h_history[key]
        if hist:
            recent = hist[-5:]
            h_wins = sum(1 for r in recent if r == ht) / len(recent)
            a_wins = sum(1 for r in recent if r == at) / len(recent)
        else:
            h_wins, a_wins = 0.33, 0.33
            
        home_h2h_wins.append(round(h_wins, 3)); away_h2h_wins.append(round(a_wins, 3))
        
        if row["result"] == "H": h2h_history[key].append(ht)
        elif row["result"] == "A": h2h_history[key].append(at)
        else: h2h_history[key].append(0)
            
        # GD Form
        h_gd, a_gd = team_gd_history[ht], team_gd_history[at]
        h_gd_avg = np.mean(h_gd[-FORM_WINDOW:]) if len(h_gd) >= FORM_WINDOW else (np.mean(h_gd) if h_gd else 0.0)
        a_gd_avg = np.mean(a_gd[-FORM_WINDOW:]) if len(a_gd) >= FORM_WINDOW else (np.mean(a_gd) if a_gd else 0.0)
        
        home_gd_form.append(round(h_gd_avg, 3)); away_gd_form.append(round(a_gd_avg, 3))
        
        team_gd_history[ht].append(row["home_goals"] - row["away_goals"])
        team_gd_history[at].append(row["away_goals"] - row["home_goals"])
        
    df["home_h2h_win_rate"] = home_h2h_wins
    df["away_h2h_win_rate"] = away_h2h_wins
    df["home_gd_form"] = home_gd_form
    df["away_gd_form"] = away_gd_form
    
    # 6. Europe features
    logger.info("  Computing Europe features...")
    team_europe_dates = defaultdict(list)
    for fix in euro_fixtures:
        match_date = pd.to_datetime(fix["date"]).tz_localize(None)
        team_europe_dates[fix["home_team_id"]].append(match_date)
        team_europe_dates[fix["away_team_id"]].append(match_date)
        
    for tid in team_europe_dates:
        team_europe_dates[tid] = sorted(team_europe_dates[tid])
        
    name_to_id = {}
    for fix in euro_fixtures:
        name_to_id[fix["home_team"]] = fix["home_team_id"]
        name_to_id[fix["away_team"]] = fix["away_team_id"]
        
    home_played_europe, away_played_europe = [], []
    for idx, row in df.iterrows():
        match_date = row["date"]
        home_id = name_to_id.get(row["home_team"])
        away_id = name_to_id.get(row["away_team"])
        
        h_europe = played_europe_in_last_7_days(home_id, match_date, team_europe_dates) if home_id else 0
        a_europe = played_europe_in_last_7_days(away_id, match_date, team_europe_dates) if away_id else 0
        
        home_played_europe.append(h_europe)
        away_played_europe.append(a_europe)
        
    df["home_played_europe"] = home_played_europe
    df["away_played_europe"] = away_played_europe
    
    # 7. Advanced Stats Rolling Averages
    logger.info("  Computing advanced stats rolling averages...")
    
    # First, attach raw stats to the dataframe
    for stat in STAT_TYPES:
        col = clean_col_name(stat)
        df[f"home_{col}"] = np.nan
        df[f"away_{col}"] = np.nan
        
    for idx, row in df.iterrows():
        fid = str(int(row["fixture_id"]))
        entry = stats_cache.get(fid, {})
        
        if entry and len(entry) >= 2:
            hs, aws = match_team_stats(entry, row["home_team"], row["away_team"])
            for sn in STAT_TYPES:
                cn = clean_col_name(sn)
                df.at[idx, f"home_{cn}"] = hs.get(sn)
                df.at[idx, f"away_{cn}"] = aws.get(sn)
                
    # Create team-match records for rolling calculation
    records = []
    for idx, row in df.iterrows():
        fid, date, league, season = row["fixture_id"], row["date"], row["league_name"], row["season"]
        
        home_rec = {"fixture_id": fid, "date": date, "team": row["home_team"], "league": league, "season": season}
        away_rec = {"fixture_id": fid, "date": date, "team": row["away_team"], "league": league, "season": season}
        
        for stat in ROLLING_STATS:
            home_rec[stat] = row.get(f"home_{stat}")
            away_rec[stat] = row.get(f"away_{stat}")
            
        records.extend([home_rec, away_rec])
        
    team_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    
    # Clean percentage columns
    for stat in ROLLING_STATS:
        if team_df[stat].dtype == object:
            team_df[stat] = team_df[stat].astype(str).str.rstrip('%').replace('nan', None)
            team_df[stat] = pd.to_numeric(team_df[stat], errors='coerce')
            
    # Compute rolling averages
    rolling_cols = {}
    for stat in ROLLING_STATS:
        col_name = f"{stat}_avg{ROLLING_WINDOW}"
        team_df[col_name] = (
            team_df.groupby("team")[stat]
            .transform(lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=1).mean())
        )
        rolling_cols[stat] = col_name
        
    # Merge back to main dataframe
    for stat, col in rolling_cols.items():
        df[f"home_{col}"] = np.nan
        df[f"away_{col}"] = np.nan
        
    for idx, row in df.iterrows():
        fid, home_team, away_team = row["fixture_id"], row["home_team"], row["away_team"]
        
        mask_home = (team_df["fixture_id"] == fid) & (team_df["team"] == home_team)
        mask_away = (team_df["fixture_id"] == fid) & (team_df["team"] == away_team)
        
        home_rows = team_df[mask_home]
        away_rows = team_df[mask_away]
        
        for stat, col in rolling_cols.items():
            if len(home_rows) > 0: df.at[idx, f"home_{col}"] = home_rows.iloc[0][col]
            if len(away_rows) > 0: df.at[idx, f"away_{col}"] = away_rows.iloc[0][col]
            
    # 8. Finalize dataset
    # Remove warm-up period (first 500 matches)
    WARMUP = 500
    df_final = df.iloc[WARMUP:].copy()
    
    output_cols = [
        "fixture_id", "date", "league_name", "season",
        "home_team", "away_team",
        "home_goals", "away_goals", "result",
        "home_elo", "away_elo", "elo_diff",
        "home_form_5", "away_form_5",
        "home_goals_for_avg", "away_goals_for_avg",
        "home_goals_against_avg", "away_goals_against_avg",
        "home_rest_days", "away_rest_days",
        "home_h2h_win_rate", "away_h2h_win_rate",
        "home_gd_form", "away_gd_form",
        "home_played_europe", "away_played_europe",
    ]
    
    # Add rolling stats columns
    for stat in ROLLING_STATS:
        output_cols.extend([f"home_{stat}_avg{ROLLING_WINDOW}", f"away_{stat}_avg{ROLLING_WINDOW}"])
        
    df_final = df_final[output_cols].reset_index(drop=True)
    return df_final

def main():
    logger.info("=" * 60)
    logger.info("STARTING DATASET UPDATE")
    logger.info("=" * 60)
    
    check_api_key()
    
    # 1. Load existing raw data
    raw_fixtures = load_json_cache(RAW_FIXTURES_FILE)
    existing_fixture_ids = {f["fixture_id"] for f in raw_fixtures}
    logger.info(f"Loaded {len(raw_fixtures)} existing fixtures.")
    
    # 2. Fetch new fixtures
    new_fixtures = fetch_new_fixtures(existing_fixture_ids)
    if not new_fixtures:
        logger.info("No new matches found. Dataset is up to date.")
        return
        
    logger.info(f"Found {len(new_fixtures)} new matches to add.")
    raw_fixtures.extend(new_fixtures)
    save_json_cache(raw_fixtures, RAW_FIXTURES_FILE)
    
    # 3. Update European fixtures
    euro_fixtures = load_json_cache(EURO_CACHE_FILE)
    existing_euro_ids = {f["fixture_id"] for f in euro_fixtures}
    new_euro_fixtures = fetch_new_european_fixtures(existing_euro_ids)
    
    if new_euro_fixtures:
        logger.info(f"Added {len(new_euro_fixtures)} new European matches.")
        euro_fixtures.extend(new_euro_fixtures)
        save_json_cache(euro_fixtures, EURO_CACHE_FILE)
        
    # 4. Fetch advanced stats for new fixtures
    stats_cache = load_json_cache(STATS_CACHE_FILE, {})
    new_fixture_ids = [f["fixture_id"] for f in new_fixtures]
    stats_cache = fetch_advanced_stats(new_fixture_ids, stats_cache)
    
    # 5. Recalculate all features and build final dataset
    raw_df = pd.DataFrame(raw_fixtures)
    final_df = recalculate_all_features(raw_df, euro_fixtures, stats_cache)
    
    # 6. Save final dataset
    final_df.to_csv(FINAL_DATASET_FILE, index=False)
    logger.info(f"Saved updated dataset to {FINAL_DATASET_FILE}")
    logger.info(f"New dataset shape: {final_df.shape}")
    
    logger.info("=" * 60)
    logger.info("UPDATE COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
