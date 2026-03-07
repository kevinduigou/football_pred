"""
=============================================================
  Football Odds Fetcher — Nice vs Rennes (8 March 2026)
  API: https://v3.football.api-sports.io
=============================================================

NOTE: Betclic is NOT available as a bookmaker in API-Football.
Available bookmakers for this fixture:
  10Bet, 188Bet, 1xBet, 888Sport, Bet365, Betano,
  Betfair, Marathonbet, Pinnacle, SBO, Unibet, William Hill

This script fetches odds from ALL available bookmakers and
displays them in a comparison table.
=============================================================
"""

import requests
import json
from datetime import datetime

# --------------------------------------------------
# Configuration
# --------------------------------------------------
API_KEY = "c1b837c37df33c46d475f5a67c346c22"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

FIXTURE_ID = 1387920  # Nice vs Rennes — 8 March 2026


def get_bookmakers():
    """List all available bookmakers in API-Football."""
    r = requests.get(f"{BASE_URL}/odds/bookmakers", headers=HEADERS)
    return r.json().get("response", [])


def get_bet_types():
    """List all available bet types."""
    r = requests.get(f"{BASE_URL}/odds/bets", headers=HEADERS)
    return r.json().get("response", [])


def get_fixture_odds(fixture_id, bookmaker_id=None):
    """
    Get pre-match odds for a specific fixture.
    Optionally filter by bookmaker ID.
    """
    params = {"fixture": fixture_id}
    if bookmaker_id:
        params["bookmaker"] = bookmaker_id
    r = requests.get(f"{BASE_URL}/odds", headers=HEADERS, params=params)
    return r.json()


def get_live_odds(fixture_id):
    """Get live/in-play odds for a fixture (only during the match)."""
    r = requests.get(f"{BASE_URL}/odds/live", headers=HEADERS, params={
        "fixture": fixture_id,
    })
    return r.json()


def display_match_winner_odds(odds_data):
    """Display Match Winner (1X2) odds from all bookmakers."""
    if not odds_data.get("response"):
        print("No odds available for this fixture.")
        return

    print("\n" + "=" * 70)
    print("  NICE vs RENNES — 8 March 2026 — Allianz Riviera")
    print("  Match Winner (1X2) Odds Comparison")
    print("=" * 70)
    print(f"\n  {'Bookmaker':<20} {'Nice (H)':>10} {'Draw':>10} {'Rennes (A)':>10}")
    print("  " + "-" * 52)

    all_home, all_draw, all_away = [], [], []

    for entry in odds_data["response"]:
        for bm in entry.get("bookmakers", []):
            bm_name = bm["name"]
            for bet in bm.get("bets", []):
                if bet["name"] == "Match Winner":
                    odds = {}
                    for val in bet["values"]:
                        odds[val["value"]] = float(val["odd"])

                    home = odds.get("Home", 0)
                    draw = odds.get("Draw", 0)
                    away = odds.get("Away", 0)

                    if home and draw and away:
                        all_home.append(home)
                        all_draw.append(draw)
                        all_away.append(away)
                        print(f"  {bm_name:<20} {home:>10.2f} {draw:>10.2f} {away:>10.2f}")

    if all_home:
        print("  " + "-" * 52)
        avg_h = sum(all_home) / len(all_home)
        avg_d = sum(all_draw) / len(all_draw)
        avg_a = sum(all_away) / len(all_away)
        print(f"  {'AVERAGE':<20} {avg_h:>10.2f} {avg_d:>10.2f} {avg_a:>10.2f}")

        best_h = max(all_home)
        best_d = max(all_draw)
        best_a = max(all_away)
        print(f"  {'BEST ODDS':<20} {best_h:>10.2f} {best_d:>10.2f} {best_a:>10.2f}")

        # Implied probabilities from average odds
        print("\n  Implied probabilities (from average odds):")
        total = (1/avg_h) + (1/avg_d) + (1/avg_a)
        print(f"    P(Nice Win):    {(1/avg_h)/total*100:.1f}%")
        print(f"    P(Draw):        {(1/avg_d)/total*100:.1f}%")
        print(f"    P(Rennes Win):  {(1/avg_a)/total*100:.1f}%")
        print(f"    Overround:      {(total - 1)*100:.1f}%")


def display_all_bets(odds_data, bookmaker_filter=None):
    """Display all bet types for a specific bookmaker or all."""
    if not odds_data.get("response"):
        print("No odds available.")
        return

    for entry in odds_data["response"]:
        for bm in entry.get("bookmakers", []):
            bm_name = bm["name"]
            if bookmaker_filter and bookmaker_filter.lower() not in bm_name.lower():
                continue

            print(f"\n{'=' * 60}")
            print(f"  {bm_name}")
            print(f"{'=' * 60}")

            for bet in bm.get("bets", []):
                print(f"\n  {bet['name']}:")
                for val in bet["values"]:
                    print(f"    {val['value']:<30} {val['odd']:>8}")


def display_popular_markets(odds_data):
    """Display popular betting markets from the best available bookmaker."""
    if not odds_data.get("response"):
        return

    # Use Bet365 or Unibet as reference (most popular in Europe)
    preferred = ["Bet365", "Unibet", "Pinnacle", "Betano"]

    for entry in odds_data["response"]:
        for pref_name in preferred:
            for bm in entry.get("bookmakers", []):
                if bm["name"] == pref_name:
                    print(f"\n{'=' * 60}")
                    print(f"  POPULAR MARKETS — {bm['name']}")
                    print(f"{'=' * 60}")

                    target_bets = [
                        "Match Winner",
                        "Goals Over/Under",
                        "Both Teams Score",
                        "Double Chance",
                        "Exact Score",
                        "First Half Winner",
                        "Total - Home",
                        "Total - Away",
                    ]

                    for bet in bm.get("bets", []):
                        if bet["name"] in target_bets:
                            print(f"\n  {bet['name']}:")
                            for val in bet["values"]:
                                print(f"    {val['value']:<30} {val['odd']:>8}")
                    return


# --------------------------------------------------
# Main execution
# --------------------------------------------------
if __name__ == "__main__":
    print(f"Fetching odds at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fixture ID: {FIXTURE_ID}")

    # Fetch odds
    odds_data = get_fixture_odds(FIXTURE_ID)

    # 1. Match Winner comparison across all bookmakers
    display_match_winner_odds(odds_data)

    # 2. Popular markets from best bookmaker
    display_popular_markets(odds_data)

    # 3. Check live odds (only available during the match)
    print(f"\n{'=' * 60}")
    print("  LIVE ODDS STATUS")
    print(f"{'=' * 60}")
    live = get_live_odds(FIXTURE_ID)
    if live.get("response"):
        print("  Live odds are available!")
        for entry in live["response"]:
            for bm in entry.get("bookmakers", []):
                print(f"\n  {bm['name']}:")
                for bet in bm.get("bets", []):
                    print(f"    {bet['name']}:")
                    for val in bet["values"]:
                        print(f"      {val['value']}: {val['odd']}")
    else:
        print("  No live odds (match has not started yet).")
        print("  Live odds become available during the match.")

    # API usage
    r = requests.get(f"{BASE_URL}/status", headers=HEADERS)
    status = r.json()
    print(f"\nAPI requests: {status['response']['requests']['current']}/{status['response']['requests']['limit_day']}")
