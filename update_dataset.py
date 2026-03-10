"""
update_dataset.py
-----------------
Fetches the latest finished fixtures from the API-Football API and appends them
to the existing football_matches.csv dataset, recomputing all features
(ELO, form, goals averages, rest days, H2H, goal-difference form, Europe flag)
from scratch to maintain consistency.

Usage:
    FOOTBALL_API_SPORTS=<api_key> python update_dataset.py [--from YYYY-MM-DD] [--to YYYY-MM-DD]

By default, fetches matches from the last 7 days.
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("FOOTBALL_API_SPORTS")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset", "football_matches.csv")
ELO_PATH = os.path.join(os.path.dirname(__file__), "dataset", "elo_ratings_all_teams.csv")
EUROPEAN_CACHE = os.path.join(os.path.dirname(__file__), "dataset", "european_fixtures_cache.json")

LEAGUES = {
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
}

EUROPEAN_LEAGUE_IDS = [2, 3, 848]  # UCL, UEL, UECL

# ELO parameters (must match original build_features_extended.py)
K = 20
INITIAL_ELO = 1500
HOME_ADVANTAGE = 50

FORM_WINDOW = 5
GOALS_WINDOW = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def check_api_status() -> dict:
    r = requests.get(f"{BASE_URL}/status", headers=HEADERS, timeout=15)
    data = r.json()
    resp = data.get("response", {})
    reqs = resp.get("requests", {})
    log(
        f"API status: {reqs.get('current', '?')}/{reqs.get('limit_day', '?')} requests used today"
    )
    return reqs


def fetch_fixtures(league_id: int, season: int, date_from: str, date_to: str) -> list:
    """Fetch all finished fixtures for a league/season in a date range."""
    r = requests.get(
        f"{BASE_URL}/fixtures",
        headers=HEADERS,
        params={
            "league": league_id,
            "season": season,
            "from": date_from,
            "to": date_to,
            "status": "FT",
        },
        timeout=15,
    )
    data = r.json()
    if data.get("errors"):
        log(f"  API error: {data['errors']}")
        return []
    return data.get("response", [])


def parse_fixture(f: dict, league_name: str, season: int) -> dict | None:
    """Parse a raw API fixture into a flat dict."""
    home_goals = f["goals"]["home"]
    away_goals = f["goals"]["away"]
    if home_goals is None or away_goals is None:
        return None

    if home_goals > away_goals:
        result = "H"
    elif home_goals < away_goals:
        result = "A"
    else:
        result = "D"

    return {
        "fixture_id": f["fixture"]["id"],
        "date": f["fixture"]["date"],
        "league_name": league_name,
        "season": season,
        "home_team_id": f["teams"]["home"]["id"],
        "home_team": f["teams"]["home"]["name"],
        "away_team_id": f["teams"]["away"]["id"],
        "away_team": f["teams"]["away"]["name"],
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": result,
    }


# ---------------------------------------------------------------------------
# Feature computation (mirrors build_features_extended.py logic)
# ---------------------------------------------------------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute all features from scratch on the full sorted dataframe."""
    df = df.sort_values("date").reset_index(drop=True)

    # --- ELO ---
    elo_ratings: dict[int, float] = defaultdict(lambda: INITIAL_ELO)
    home_elos, away_elos, elo_diffs = [], [], []

    for _, row in df.iterrows():
        ht = int(row["home_team_id"])
        at = int(row["away_team_id"])
        h_elo = elo_ratings[ht]
        a_elo = elo_ratings[at]
        home_elos.append(h_elo)
        away_elos.append(a_elo)
        elo_diffs.append(h_elo - a_elo)

        exp_home = 1 / (1 + 10 ** ((a_elo - (h_elo + HOME_ADVANTAGE)) / 400))
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

    # --- Form (points in last 5 matches) ---
    team_history: dict[int, list] = defaultdict(list)
    home_form_5, away_form_5 = [], []

    for _, row in df.iterrows():
        ht = int(row["home_team_id"])
        at = int(row["away_team_id"])
        h_hist = team_history[ht]
        a_hist = team_history[at]
        h_form = np.mean(h_hist[-FORM_WINDOW:]) if h_hist else 1.0
        a_form = np.mean(a_hist[-FORM_WINDOW:]) if a_hist else 1.0
        home_form_5.append(round(float(h_form), 3))
        away_form_5.append(round(float(a_form), 3))
        if row["result"] == "H":
            team_history[ht].append(3)
            team_history[at].append(0)
        elif row["result"] == "A":
            team_history[ht].append(0)
            team_history[at].append(3)
        else:
            team_history[ht].append(1)
            team_history[at].append(1)

    df["home_form_5"] = home_form_5
    df["away_form_5"] = away_form_5

    # --- Goals averages (last 5 matches) ---
    team_goals_for: dict[int, list] = defaultdict(list)
    team_goals_against: dict[int, list] = defaultdict(list)
    home_gf_avg, away_gf_avg, home_ga_avg, away_ga_avg = [], [], [], []

    for _, row in df.iterrows():
        ht = int(row["home_team_id"])
        at = int(row["away_team_id"])
        hg = row["home_goals"]
        ag = row["away_goals"]

        h_gf = team_goals_for[ht]
        h_ga = team_goals_against[ht]
        a_gf = team_goals_for[at]
        a_ga = team_goals_against[at]

        h_gf_avg = np.mean(h_gf[-GOALS_WINDOW:]) if h_gf else 1.3
        a_gf_avg = np.mean(a_gf[-GOALS_WINDOW:]) if a_gf else 1.3
        h_ga_avg = np.mean(h_ga[-GOALS_WINDOW:]) if h_ga else 1.1
        a_ga_avg = np.mean(a_ga[-GOALS_WINDOW:]) if a_ga else 1.1

        home_gf_avg.append(round(float(h_gf_avg), 3))
        away_gf_avg.append(round(float(a_gf_avg), 3))
        home_ga_avg.append(round(float(h_ga_avg), 3))
        away_ga_avg.append(round(float(a_ga_avg), 3))

        team_goals_for[ht].append(hg)
        team_goals_against[ht].append(ag)
        team_goals_for[at].append(ag)
        team_goals_against[at].append(hg)

    df["home_goals_for_avg"] = home_gf_avg
    df["away_goals_for_avg"] = away_gf_avg
    df["home_goals_against_avg"] = home_ga_avg
    df["away_goals_against_avg"] = away_ga_avg

    # --- Rest days ---
    team_last_match: dict[int, pd.Timestamp] = {}
    home_rest, away_rest = [], []

    for _, row in df.iterrows():
        ht = int(row["home_team_id"])
        at = int(row["away_team_id"])
        match_date = row["date"]

        h_rest = min((match_date - team_last_match[ht]).days, 30) if ht in team_last_match else 7
        a_rest = min((match_date - team_last_match[at]).days, 30) if at in team_last_match else 7

        home_rest.append(h_rest)
        away_rest.append(a_rest)
        team_last_match[ht] = match_date
        team_last_match[at] = match_date

    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest

    # --- H2H win rate and goal-difference form ---
    h2h_history: dict[tuple, list] = defaultdict(list)
    team_gd_history: dict[int, list] = defaultdict(list)
    home_h2h_wins, away_h2h_wins, home_gd_form, away_gd_form = [], [], [], []

    for _, row in df.iterrows():
        ht = int(row["home_team_id"])
        at = int(row["away_team_id"])
        key = (min(ht, at), max(ht, at))
        hist = h2h_history[key]

        if hist:
            recent = hist[-5:]
            h_wins = sum(1 for r in recent if r == ht) / len(recent)
            a_wins = sum(1 for r in recent if r == at) / len(recent)
        else:
            h_wins = 0.33
            a_wins = 0.33

        home_h2h_wins.append(round(h_wins, 3))
        away_h2h_wins.append(round(a_wins, 3))

        h_gd = team_gd_history[ht]
        a_gd = team_gd_history[at]
        h_gd_avg = np.mean(h_gd[-FORM_WINDOW:]) if h_gd else 0.0
        a_gd_avg = np.mean(a_gd[-FORM_WINDOW:]) if a_gd else 0.0
        home_gd_form.append(round(float(h_gd_avg), 3))
        away_gd_form.append(round(float(a_gd_avg), 3))

        if row["result"] == "H":
            h2h_history[key].append(ht)
        elif row["result"] == "A":
            h2h_history[key].append(at)
        else:
            h2h_history[key].append(0)

        team_gd_history[ht].append(row["home_goals"] - row["away_goals"])
        team_gd_history[at].append(row["away_goals"] - row["home_goals"])

    df["home_h2h_win_rate"] = home_h2h_wins
    df["away_h2h_win_rate"] = away_h2h_wins
    df["home_gd_form"] = home_gd_form
    df["away_gd_form"] = away_gd_form

    return df


def add_europe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add home_played_europe / away_played_europe binary columns.
    Uses the cached European fixtures if available; otherwise sets 0.
    """
    import json

    if not os.path.exists(EUROPEAN_CACHE):
        log("European fixtures cache not found – setting home/away_played_europe = 0")
        df["home_played_europe"] = 0
        df["away_played_europe"] = 0
        return df

    with open(EUROPEAN_CACHE) as f:
        european_fixtures = json.load(f)

    # Build team -> sorted list of European match dates
    team_europe_dates: dict[int, list] = defaultdict(list)
    for fix in european_fixtures:
        date = pd.to_datetime(fix["date"], utc=True)
        team_europe_dates[fix["home_team_id"]].append(date)
        team_europe_dates[fix["away_team_id"]].append(date)

    # Build name -> id mapping (from European data + dataset)
    name_to_id: dict[str, int] = {}
    for fix in european_fixtures:
        name_to_id[fix["home_team"]] = fix["home_team_id"]
        name_to_id[fix["away_team"]] = fix["away_team_id"]

    # Also populate from the dataset itself (home_team_id column)
    for _, row in df.iterrows():
        name_to_id[row["home_team"]] = int(row["home_team_id"])
        name_to_id[row["away_team"]] = int(row["away_team_id"])

    def played_europe_in_last_7(team_id: int, match_date: pd.Timestamp) -> int:
        dates = team_europe_dates.get(team_id, [])
        for d in dates:
            delta = (match_date - d).days
            if 0 < delta <= 7:
                return 1
        return 0

    home_played_europe, away_played_europe = [], []
    for _, row in df.iterrows():
        match_date = row["date"]
        home_id = name_to_id.get(row["home_team"])
        away_id = name_to_id.get(row["away_team"])
        h_eur = played_europe_in_last_7(home_id, match_date) if home_id else 0
        a_eur = played_europe_in_last_7(away_id, match_date) if away_id else 0
        home_played_europe.append(h_eur)
        away_played_europe.append(a_eur)

    df["home_played_europe"] = home_played_europe
    df["away_played_europe"] = away_played_europe
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Update football dataset with latest results")
    parser.add_argument(
        "--from",
        dest="date_from",
        default=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--to",
        dest="date_to",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    if not API_KEY:
        log("ERROR: FOOTBALL_API_SPORTS environment variable not set.")
        sys.exit(1)

    log("=" * 60)
    log("FOOTBALL DATASET UPDATE")
    log("=" * 60)
    log(f"Date range: {args.date_from} → {args.date_to}")

    check_api_status()

    # --- Load existing dataset ---
    log(f"\nLoading existing dataset from {DATASET_PATH}")
    df_existing = pd.read_csv(DATASET_PATH)
    df_existing["date"] = pd.to_datetime(df_existing["date"], utc=True)
    existing_ids = set(df_existing["fixture_id"].astype(int).tolist())
    log(f"Existing dataset: {len(df_existing)} matches (last date: {df_existing['date'].max().date()})")

    # --- Fetch new fixtures ---
    log(f"\nFetching new fixtures ({args.date_from} → {args.date_to})...")
    new_rows = []
    for league_id, league_name in LEAGUES.items():
        fixtures = fetch_fixtures(league_id, 2025, args.date_from, args.date_to)
        added = 0
        for f in fixtures:
            fid = f["fixture"]["id"]
            if fid in existing_ids:
                continue
            parsed = parse_fixture(f, league_name, 2025)
            if parsed:
                new_rows.append(parsed)
                added += 1
        log(f"  {league_name}: {len(fixtures)} fetched, {added} new")
        time.sleep(0.4)

    if not new_rows:
        log("\nNo new matches found. Dataset is already up to date.")
        return

    log(f"\nNew matches to add: {len(new_rows)}")
    for r in new_rows:
        log(f"  [{r['league_name']}] {r['date'][:10]}: {r['home_team']} {r['home_goals']}-{r['away_goals']} {r['away_team']} (FID:{r['fixture_id']})")

    # --- We need team IDs for the existing dataset rows too ---
    # The existing dataset may not have home_team_id / away_team_id columns.
    # We'll build a name->id map from the new rows and from the European cache.
    log("\nBuilding team name → ID mapping...")
    name_to_id: dict[str, int] = {}
    for row in new_rows:
        name_to_id[row["home_team"]] = row["home_team_id"]
        name_to_id[row["away_team"]] = row["away_team_id"]

    # Try to enrich from European cache
    import json
    if os.path.exists(EUROPEAN_CACHE):
        with open(EUROPEAN_CACHE) as f:
            euro_cache = json.load(f)
        for fix in euro_cache:
            name_to_id[fix["home_team"]] = fix["home_team_id"]
            name_to_id[fix["away_team"]] = fix["away_team_id"]

    # Add team IDs to existing dataset if not present
    if "home_team_id" not in df_existing.columns:
        df_existing["home_team_id"] = df_existing["home_team"].map(name_to_id)
        df_existing["away_team_id"] = df_existing["away_team"].map(name_to_id)

    # For any still-missing IDs, assign a synthetic negative ID based on team name hash
    # (so ELO tracking still works consistently)
    all_teams_in_existing = set(df_existing["home_team"].tolist() + df_existing["away_team"].tolist())
    for team in all_teams_in_existing:
        if team not in name_to_id:
            name_to_id[team] = abs(hash(team)) % 1_000_000 + 100_000
    df_existing["home_team_id"] = df_existing["home_team_id"].fillna(
        df_existing["home_team"].map(name_to_id)
    )
    df_existing["away_team_id"] = df_existing["away_team_id"].fillna(
        df_existing["away_team"].map(name_to_id)
    )

    # --- Merge new rows ---
    df_new = pd.DataFrame(new_rows)
    df_new["date"] = pd.to_datetime(df_new["date"], utc=True)

    # Keep only columns needed for feature computation
    base_cols = [
        "fixture_id", "date", "league_name", "season",
        "home_team_id", "home_team",
        "away_team_id", "away_team",
        "home_goals", "away_goals", "result",
    ]
    df_combined = pd.concat(
        [df_existing[base_cols], df_new[base_cols]], ignore_index=True
    )
    df_combined = df_combined.drop_duplicates(subset=["fixture_id"]).sort_values("date").reset_index(drop=True)
    df_combined["home_team_id"] = df_combined["home_team_id"].astype(int)
    df_combined["away_team_id"] = df_combined["away_team_id"].astype(int)

    log(f"\nCombined dataset before feature computation: {len(df_combined)} matches")

    # --- Recompute all features ---
    log("Recomputing all features (ELO, form, goals, rest, H2H, GD)...")
    df_featured = compute_features(df_combined)

    # --- Add European competition feature ---
    log("Adding European competition features...")
    df_featured = add_europe_features(df_featured)

    # --- Final output columns ---
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
    df_out = df_featured[output_cols].copy()

    # --- Save ---
    df_out.to_csv(DATASET_PATH, index=False)
    log(f"\nDataset saved: {len(df_out)} matches → {DATASET_PATH}")

    # --- Show newly added matches ---
    new_fids = {r["fixture_id"] for r in new_rows}
    df_added = df_out[df_out["fixture_id"].isin(new_fids)].sort_values("date")
    log(f"\n{'='*60}")
    log(f"NEW MATCHES ADDED ({len(df_added)})")
    log(f"{'='*60}")
    for _, row in df_added.iterrows():
        log(
            f"  [{row['league_name']}] {str(row['date'])[:10]}: "
            f"{row['home_team']} {int(row['home_goals'])}-{int(row['away_goals'])} {row['away_team']} "
            f"({row['result']}) | ELO diff: {row['elo_diff']:.0f}"
        )

    # --- Show Nice vs Rennes specifically ---
    nice_rennes = df_added[
        (df_added["home_team"].str.contains("Nice", na=False) & df_added["away_team"].str.contains("Rennes", na=False)) |
        (df_added["home_team"].str.contains("Rennes", na=False) & df_added["away_team"].str.contains("Nice", na=False))
    ]
    if not nice_rennes.empty:
        log(f"\n{'='*60}")
        log("NICE vs RENNES DETAILS")
        log(f"{'='*60}")
        for _, row in nice_rennes.iterrows():
            log(f"  Date:         {str(row['date'])[:10]}")
            log(f"  Match:        {row['home_team']} {int(row['home_goals'])}-{int(row['away_goals'])} {row['away_team']}")
            log(f"  Result:       {row['result']}")
            log(f"  Home ELO:     {row['home_elo']:.1f}")
            log(f"  Away ELO:     {row['away_elo']:.1f}")
            log(f"  ELO diff:     {row['elo_diff']:.1f}")
            log(f"  Home form 5:  {row['home_form_5']}")
            log(f"  Away form 5:  {row['away_form_5']}")
            log(f"  Home GD form: {row['home_gd_form']}")
            log(f"  Away GD form: {row['away_gd_form']}")
            log(f"  Home played Europe: {row['home_played_europe']}")
            log(f"  Away played Europe: {row['away_played_europe']}")

    log(f"\nDone. Total matches in dataset: {len(df_out)}")


if __name__ == "__main__":
    main()
