#!/usr/bin/env python3
"""
Value Bets Detector
===================

Fetches upcoming matches for the next weekend across the 5 major European
leagues, computes win/draw/loss probabilities with the XGBoost v4 model,
retrieves bookmaker odds from API-Football, and identifies **value bets**
where the model's estimated probability exceeds the bookmaker's implied
probability by a configurable threshold.

Requirements:
    pip install pandas numpy requests tabulate scikit-learn xgboost

Usage:
    export FOOTBALL_API_SPORTS="your_api_key"
    python3 value_bets.py                       # default 5% threshold, 7 days
    python3 value_bets.py --threshold 0.10      # 10% edge required
    python3 value_bets.py --days 3              # only next 3 days
    python3 value_bets.py --output results.csv  # save full table to CSV

The script can also be imported as a module:
    from value_bets import find_value_bets
    bets = find_value_bets(threshold=0.05, days_ahead=7)
"""

import os
import sys
import json
import time
import pickle
import logging
import argparse
import warnings
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/value_bets.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("FOOTBALL_API_SPORTS", "")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

LEAGUES = {
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
}

# Current season (API-Football uses the year the season *starts*)
CURRENT_SEASON = 2024

# Paths (relative to the repo root so the script works from anywhere)
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "dataset"
DATASET_FILE = DATA_DIR / "football_matches_v4.csv"
STATS_CACHE_FILE = DATA_DIR / "advanced_stats_cache.json"
EURO_CACHE_FILE = DATA_DIR / "european_fixtures_cache.json"
MODEL_FILE = REPO_ROOT / "models" / "xgb_football_model_v4_advanced.pkl"

# Feature-engineering constants (must match the training pipeline)
K = 20
INITIAL_ELO = 1500
HOME_ADVANTAGE = 50
FORM_WINDOW = 5

ROLL_STATS = [
    "shots_on_goal", "total_shots", "shots_insidebox",
    "ball_possession", "total_passes", "passes_pct",
    "corner_kicks", "fouls", "expected_goals",
]

RAW_STAT_MAP = {
    "shots_on_goal":   "Shots on Goal",
    "total_shots":     "Total Shots",
    "shots_insidebox": "Shots insidebox",
    "ball_possession": "Ball Possession",
    "total_passes":    "Total passes",
    "passes_pct":      "Passes %",
    "corner_kicks":    "Corner Kicks",
    "fouls":           "Fouls",
    "expected_goals":  "expected_goals",
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def _api_get(endpoint: str, params: dict | None = None) -> dict:
    """Perform a GET request to API-Football with error handling."""
    try:
        r = requests.get(
            f"{BASE_URL}/{endpoint}",
            headers=HEADERS,
            params=params or {},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        logger.error("API request failed for /%s: %s", endpoint, exc)
        return {}


def check_api_status() -> int:
    """Return the number of remaining API requests for today."""
    data = _api_get("status")
    req = data.get("response", {}).get("requests", {})
    cur = req.get("current", 0)
    lim = req.get("limit_day", 7500)
    remaining = lim - cur
    logger.info("API quota: %d/%d used (%d remaining)", cur, lim, remaining)
    return remaining


# ---------------------------------------------------------------------------
# Upcoming fixtures
# ---------------------------------------------------------------------------
def get_upcoming_fixtures(days_ahead: int = 7) -> list[dict]:
    """Fetch upcoming (Not Started) fixtures for the top-5 leagues."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    future = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    logger.info("Fetching fixtures from %s to %s ...", today, future)

    upcoming: list[dict] = []
    for lid, lname in LEAGUES.items():
        data = _api_get("fixtures", {
            "league": lid,
            "season": CURRENT_SEASON,
            "from": today,
            "to": future,
            "status": "NS",
        })
        for f in data.get("response", []):
            upcoming.append({
                "fixture_id": f["fixture"]["id"],
                "date": pd.to_datetime(f["fixture"]["date"]).tz_localize(None),
                "league_id": lid,
                "league_name": lname,
                "home_team_id": f["teams"]["home"]["id"],
                "home_team": f["teams"]["home"]["name"],
                "away_team_id": f["teams"]["away"]["id"],
                "away_team": f["teams"]["away"]["name"],
            })
        logger.info("  %s: %d upcoming", lname,
                     sum(1 for u in upcoming if u["league_name"] == lname))
        time.sleep(0.3)

    upcoming.sort(key=lambda m: m["date"])
    return upcoming


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------
def get_match_winner_odds(fixture_id: int) -> dict | None:
    """
    Return average + best Match Winner (1X2) odds across all bookmakers.

    Returns dict with keys:
        avg_home, avg_draw, avg_away   – average decimal odds
        best_home, best_draw, best_away – best (highest) decimal odds
        bookmakers_count               – number of bookmakers
    or None when no odds are available.
    """
    data = _api_get("odds", {"fixture": fixture_id})
    if not data.get("response"):
        return None

    homes, draws, aways = [], [], []
    for entry in data["response"]:
        for bm in entry.get("bookmakers", []):
            for bet in bm.get("bets", []):
                if bet["name"] == "Match Winner":
                    odds_map = {v["value"]: float(v["odd"]) for v in bet["values"]}
                    h, d, a = odds_map.get("Home"), odds_map.get("Draw"), odds_map.get("Away")
                    if h and d and a:
                        homes.append(h); draws.append(d); aways.append(a)

    if not homes:
        return None

    return {
        "avg_home": np.mean(homes), "avg_draw": np.mean(draws), "avg_away": np.mean(aways),
        "best_home": max(homes), "best_draw": max(draws), "best_away": max(aways),
        "bookmakers_count": len(homes),
    }


# ---------------------------------------------------------------------------
# Team state reconstruction (ELO, form, rolling stats)
# ---------------------------------------------------------------------------
def build_team_states() -> dict:
    """
    Replay the full match history to obtain the *current* state of every team:
    ELO rating, recent form, goals averages, head-to-head, goal-difference
    form, and rolling advanced-stats averages.
    """
    logger.info("Rebuilding team states from %s ...", DATASET_FILE.name)

    df = pd.read_csv(DATASET_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    cache: dict = {}
    if STATS_CACHE_FILE.exists():
        with open(STATS_CACHE_FILE) as fh:
            cache = json.load(fh)

    # Build Europe lookup for the played_europe feature
    euro_dates: dict[str, list] = defaultdict(list)
    if EURO_CACHE_FILE.exists():
        with open(EURO_CACHE_FILE) as fh:
            euro_fixtures = json.load(fh)
        name_to_id: dict[str, int] = {}
        for ef in euro_fixtures:
            dt = pd.to_datetime(ef["date"]).tz_localize(None)
            euro_dates[ef["home_team"]].append(dt)
            euro_dates[ef["away_team"]].append(dt)
            name_to_id[ef["home_team"]] = ef["home_team_id"]
            name_to_id[ef["away_team"]] = ef["away_team_id"]
        for t in euro_dates:
            euro_dates[t].sort()

    elo = defaultdict(lambda: INITIAL_ELO)
    history = defaultdict(list)
    goals_for = defaultdict(list)
    goals_against = defaultdict(list)
    last_match: dict = {}
    h2h = defaultdict(list)
    gd_hist = defaultdict(list)
    adv_stats = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        hg, ag = row["home_goals"], row["away_goals"]
        fid = str(int(row["fixture_id"]))

        # Advanced stats from cache
        entry = cache.get(fid, {})
        if entry and len(entry) >= 2:
            tids = list(entry.keys())
            t1, t2 = entry[tids[0]], entry[tids[1]]
            n1, n2 = t1.get("team_name", ""), t2.get("team_name", "")
            if n1.lower() in ht.lower() or ht.lower() in n1.lower():
                hs, aws = t1, t2
            elif n2.lower() in ht.lower() or ht.lower() in n2.lower():
                hs, aws = t2, t1
            else:
                hs, aws = t1, t2
            for sk, rk in RAW_STAT_MAP.items():
                for team, stats in [(ht, hs), (at, aws)]:
                    v = stats.get(rk)
                    if v is not None:
                        try:
                            adv_stats[team][sk].append(float(str(v).replace("%", "").strip()))
                        except (ValueError, TypeError):
                            pass

        # ELO update
        h_elo, a_elo = elo[ht], elo[at]
        exp_h = 1 / (1 + 10 ** ((a_elo - (h_elo + HOME_ADVANTAGE)) / 400))
        key = tuple(sorted([ht, at]))
        if row["result"] == "H":
            ah, aa = 1.0, 0.0
            history[ht].append(3); history[at].append(0)
            h2h[key].append(ht)
        elif row["result"] == "A":
            ah, aa = 0.0, 1.0
            history[ht].append(0); history[at].append(3)
            h2h[key].append(at)
        else:
            ah, aa = 0.5, 0.5
            history[ht].append(1); history[at].append(1)
            h2h[key].append("D")
        elo[ht] += K * (ah - exp_h)
        elo[at] += K * (aa - (1 - exp_h))
        goals_for[ht].append(hg); goals_against[ht].append(ag)
        goals_for[at].append(ag); goals_against[at].append(hg)
        last_match[ht] = row["date"]; last_match[at] = row["date"]
        gd_hist[ht].append(hg - ag); gd_hist[at].append(ag - hg)

    logger.info("  %d teams tracked.", len(elo))
    return {
        "elo": elo, "history": history,
        "goals_for": goals_for, "goals_against": goals_against,
        "last_match": last_match, "h2h": h2h, "gd_hist": gd_hist,
        "adv_stats": adv_stats, "euro_dates": euro_dates,
    }


# ---------------------------------------------------------------------------
# Feature extraction for a single upcoming match
# ---------------------------------------------------------------------------
def _avg5(lst: list) -> float:
    vals = [v for v in lst[-FORM_WINDOW:] if v is not None]
    return round(float(np.mean(vals)), 3) if vals else np.nan


def extract_features(match: dict, state: dict) -> dict | None:
    """Build the 33-feature vector expected by the v4 model."""
    ht, at = match["home_team"], match["away_team"]
    md = match["date"]

    if ht not in state["elo"] or at not in state["elo"]:
        logger.warning("Unknown team(s): %s / %s — skipping.", ht, at)
        return None

    # H2H
    h2h_key = tuple(sorted([ht, at]))
    h2h_recent = state["h2h"][h2h_key][-5:]
    h_h2h = sum(1 for r in h2h_recent if r == ht) / len(h2h_recent) if h2h_recent else 0.33
    a_h2h = sum(1 for r in h2h_recent if r == at) / len(h2h_recent) if h2h_recent else 0.33

    # Rest days
    def _rest(team):
        lm = state["last_match"].get(team)
        if lm is None:
            return 7
        lm = lm.tz_localize(None) if hasattr(lm, "tz_localize") and lm.tzinfo else lm
        return min((md - lm).days, 30)

    # Europe feature
    def _played_europe(team):
        dates = state["euro_dates"].get(team, [])
        window_start = md - timedelta(days=7)
        return int(any(window_start <= d < md for d in dates))

    features = {
        "home_elo":               round(state["elo"][ht], 1),
        "away_elo":               round(state["elo"][at], 1),
        "elo_diff":               round(state["elo"][ht] - state["elo"][at], 1),
        "home_form_5":            _avg5(state["history"][ht]) if state["history"][ht] else 1.0,
        "away_form_5":            _avg5(state["history"][at]) if state["history"][at] else 1.0,
        "home_goals_for_avg":     _avg5(state["goals_for"][ht]) if state["goals_for"][ht] else 1.3,
        "away_goals_for_avg":     _avg5(state["goals_for"][at]) if state["goals_for"][at] else 1.3,
        "home_goals_against_avg": _avg5(state["goals_against"][ht]) if state["goals_against"][ht] else 1.1,
        "away_goals_against_avg": _avg5(state["goals_against"][at]) if state["goals_against"][at] else 1.1,
        "home_rest_days":         _rest(ht),
        "away_rest_days":         _rest(at),
        "home_h2h_win_rate":      round(h_h2h, 3),
        "away_h2h_win_rate":      round(a_h2h, 3),
        "home_gd_form":           _avg5(state["gd_hist"][ht]) if state["gd_hist"][ht] else 0.0,
        "away_gd_form":           _avg5(state["gd_hist"][at]) if state["gd_hist"][at] else 0.0,
        "home_played_europe":     _played_europe(ht),
        "away_played_europe":     _played_europe(at),
    }

    # Rolling advanced stats (only those the model actually uses)
    for sk in ROLL_STATS:
        features[f"home_{sk}_avg5"] = _avg5(state["adv_stats"][ht][sk])
        features[f"away_{sk}_avg5"] = _avg5(state["adv_stats"][at][sk])

    return features


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def find_value_bets(
    threshold: float = 0.05,
    days_ahead: int = 7,
    use_best_odds: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Main entry point.  Returns (all_results, value_bets).

    Parameters
    ----------
    threshold : float
        Minimum edge (model_prob - implied_prob) to flag a value bet.
    days_ahead : int
        How many days ahead to scan for fixtures.
    use_best_odds : bool
        If True, compare against the best (highest) bookmaker odds instead
        of the average.
    """
    check_api_status()

    # Load model
    logger.info("Loading model from %s ...", MODEL_FILE.name)
    with open(MODEL_FILE, "rb") as fh:
        model = pickle.load(fh)
    feature_order = list(model.feature_names_in_)

    # Build team states
    state = build_team_states()

    # Fetch upcoming fixtures
    upcoming = get_upcoming_fixtures(days_ahead)
    if not upcoming:
        logger.info("No upcoming matches found.")
        return [], []

    logger.info("Processing %d upcoming matches ...", len(upcoming))

    all_results: list[dict] = []
    value_bets: list[dict] = []

    for i, match in enumerate(upcoming):
        # Features
        features = extract_features(match, state)
        if features is None:
            continue

        # Filter to only the features the model expects
        model_features = {k: features[k] for k in feature_order if k in features}
        if len(model_features) < len(feature_order):
            missing = set(feature_order) - set(model_features.keys())
            for m in missing:
                model_features[m] = np.nan

        X = pd.DataFrame([model_features])[feature_order]
        proba = model.predict_proba(X)[0]
        p_h, p_d, p_a = float(proba[0]), float(proba[1]), float(proba[2])

        # Odds
        odds = get_match_winner_odds(match["fixture_id"])
        time.sleep(0.2)

        if odds is None:
            logger.debug("No odds for %s vs %s", match["home_team"], match["away_team"])
            # Still record the prediction even without odds
            all_results.append({
                "Date": match["date"].strftime("%a %d %b %H:%M"),
                "League": match["league_name"],
                "Match": f"{match['home_team']} vs {match['away_team']}",
                "P(H)": f"{p_h*100:.1f}%",
                "P(D)": f"{p_d*100:.1f}%",
                "P(A)": f"{p_a*100:.1f}%",
                "Odds H": "-",
                "Odds D": "-",
                "Odds A": "-",
                "EV(H)": "-",
                "EV(D)": "-",
                "EV(A)": "-",
                "Value": "",
            })
            continue

        # Choose average or best odds
        prefix = "best" if use_best_odds else "avg"
        o_h = odds[f"{prefix}_home"]
        o_d = odds[f"{prefix}_draw"]
        o_a = odds[f"{prefix}_away"]

        # Implied probabilities (overround-adjusted)
        raw_h, raw_d, raw_a = 1 / o_h, 1 / o_d, 1 / o_a
        overround = raw_h + raw_d + raw_a
        imp_h, imp_d, imp_a = raw_h / overround, raw_d / overround, raw_a / overround

        # Expected Value: EV = P_model * Odds - 1
        ev_h = p_h * o_h - 1
        ev_d = p_d * o_d - 1
        ev_a = p_a * o_a - 1

        # Edge = model_prob - implied_prob
        edge_h = p_h - imp_h
        edge_d = p_d - imp_d
        edge_a = p_a - imp_a

        # Detect value bets
        flags = []
        for label, edge, ev, odds_val, model_p, imp_p, team_name in [
            ("H", edge_h, ev_h, o_h, p_h, imp_h, match["home_team"]),
            ("D", edge_d, ev_d, o_d, p_d, imp_d, "Draw"),
            ("A", edge_a, ev_a, o_a, p_a, imp_a, match["away_team"]),
        ]:
            if edge >= threshold and ev > 0:
                flags.append(label)
                value_bets.append({
                    "Date": match["date"].strftime("%a %d %b %H:%M"),
                    "Match": f"{match['home_team']} vs {match['away_team']}",
                    "Bet": f"{team_name} ({label})",
                    "Odds": f"{odds_val:.2f}",
                    "Model": f"{model_p*100:.1f}%",
                    "Implied": f"{imp_p*100:.1f}%",
                    "Edge": f"+{edge*100:.1f}%",
                    "EV": f"+{ev*100:.1f}%",
                    "Bookmakers": odds["bookmakers_count"],
                })

        flag_str = " ".join(f"[{f}]" for f in flags) if flags else ""

        all_results.append({
            "Date": match["date"].strftime("%a %d %b %H:%M"),
            "League": match["league_name"],
            "Match": f"{match['home_team']} vs {match['away_team']}",
            "P(H)": f"{p_h*100:.1f}%",
            "P(D)": f"{p_d*100:.1f}%",
            "P(A)": f"{p_a*100:.1f}%",
            "Odds H": f"{o_h:.2f}",
            "Odds D": f"{o_d:.2f}",
            "Odds A": f"{o_a:.2f}",
            "EV(H)": f"{ev_h*100:+.1f}%",
            "EV(D)": f"{ev_d*100:+.1f}%",
            "EV(A)": f"{ev_a*100:+.1f}%",
            "Value": flag_str,
        })

        if (i + 1) % 10 == 0:
            logger.info("  %d/%d matches processed ...", i + 1, len(upcoming))

    return all_results, value_bets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Detect value bets by comparing XGBoost v4 model probabilities "
                    "with bookmaker odds for upcoming matches.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="Minimum edge (model_prob - implied_prob) to flag a value bet (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Number of days ahead to scan for fixtures (default: 7)",
    )
    parser.add_argument(
        "--best-odds", action="store_true",
        help="Use the best (highest) bookmaker odds instead of the average",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save the full results table to a CSV file",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("  VALUE BETS DETECTOR — %s", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    logger.info("  Threshold: %.1f%%  |  Window: %d days  |  Odds: %s",
                args.threshold * 100, args.days, "best" if args.best_odds else "average")
    logger.info("=" * 70)

    all_results, value_bets = find_value_bets(
        threshold=args.threshold,
        days_ahead=args.days,
        use_best_odds=args.best_odds,
    )

    # ── Display all matches ──────────────────────────────────────────────
    print("\n" + "=" * 110)
    print(f"  ALL UPCOMING MATCHES ({len(all_results)} found)")
    print("=" * 110)
    if all_results:
        print(tabulate(pd.DataFrame(all_results), headers="keys",
                        tablefmt="pipe", showindex=False))
    else:
        print("  No upcoming matches found.")

    # ── Display value bets ───────────────────────────────────────────────
    print("\n" + "=" * 110)
    print(f"  VALUE BETS (edge >= {args.threshold*100:.1f}% and EV > 0)")
    print("=" * 110)
    if value_bets:
        vb_sorted = sorted(value_bets,
                           key=lambda x: float(x["EV"].replace("%", "").replace("+", "")),
                           reverse=True)
        print(tabulate(pd.DataFrame(vb_sorted), headers="keys",
                        tablefmt="pipe", showindex=False))
        print(f"\n  Total value bets found: {len(vb_sorted)}")
    else:
        print(f"  No value bets found with edge >= {args.threshold*100:.1f}%.")

    # ── Save to CSV ──────────────────────────────────────────────────────
    if args.output and all_results:
        pd.DataFrame(all_results).to_csv(args.output, index=False)
        logger.info("Full results saved to %s", args.output)

    if value_bets:
        vb_file = f"/tmp/value_bets_{datetime.utcnow().strftime('%Y%m%d')}.csv"
        pd.DataFrame(value_bets).to_csv(vb_file, index=False)
        logger.info("Value bets saved to %s", vb_file)


if __name__ == "__main__":
    main()
