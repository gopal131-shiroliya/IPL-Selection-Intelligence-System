from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cricsheet_loader import load_ipl_dataset
from feature_engineering import build_player_features
from matchup_analysis import build_batter_bowler_matchups
from scoring import build_best_xi, score_players
from valuation import estimate_player_value
from venue_analysis import build_venue_player_scores


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a selector-style IPL player recommendation report.")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="*",
        help="Season start years to analyse, for example: --seasons 2023 2024 2025",
    )
    parser.add_argument(
        "--include-current-season",
        action="store_true",
        help="Include the latest in-progress season if it exists in the Cricsheet download.",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Number of top overall players to print.")
    return parser.parse_args()


def print_section(title: str, df: pd.DataFrame, columns: list[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(df.loc[:, columns].to_string(index=False))


def default_season_window(matches: pd.DataFrame) -> list[int]:
    available = sorted(matches["season_year"].unique().tolist())
    if not available:
        return []
    return available[-3:]


def export_outputs(
    features: pd.DataFrame,
    scored: pd.DataFrame,
    best_xi: pd.DataFrame,
    valued: pd.DataFrame,
    venue_scores: pd.DataFrame,
    matchups: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    features.to_csv(OUTPUT_DIR / "player_features.csv", index=False)
    scored.to_csv(OUTPUT_DIR / "selector_rankings.csv", index=False)
    best_xi.to_csv(OUTPUT_DIR / "best_xi.csv", index=False)
    valued.to_csv(OUTPUT_DIR / "auction_value_rankings.csv", index=False)
    venue_scores.to_csv(OUTPUT_DIR / "venue_specialists.csv", index=False)
    matchups.to_csv(OUTPUT_DIR / "batter_bowler_matchups.csv", index=False)


def main() -> None:
    args = parse_args()

    raw_deliveries, raw_fielding, all_matches = load_ipl_dataset(
        seasons=args.seasons,
        include_incomplete_season=args.include_current_season,
    )

    selected_seasons = args.seasons or default_season_window(all_matches)
    deliveries, fielding, matches = load_ipl_dataset(
        seasons=selected_seasons,
        include_incomplete_season=args.include_current_season,
    )

    player_features = build_player_features(deliveries, fielding)
    scored = score_players(player_features)
    best_xi = build_best_xi(scored)
    valued = estimate_player_value(scored)
    venue_scores = build_venue_player_scores(deliveries)
    matchups = build_batter_bowler_matchups(deliveries)
    export_outputs(player_features, scored, best_xi, valued, venue_scores, matchups)

    match_count = matches["match_id"].nunique()
    latest_date = matches["match_date"].max().date()

    print("IPL Selection Intelligence Report")
    print("================================")
    print(f"Seasons analysed: {', '.join(str(season) for season in selected_seasons)}")
    print(f"Matches analysed: {match_count}")
    print(f"Latest match date in analysis: {latest_date}")
    print(f"CSV exports written to: {OUTPUT_DIR}")

    summary_columns = [
        "player_name",
        "role",
        "matches_played",
        "runs",
        "wickets",
        "strike_rate",
        "economy",
        "final_selector_score",
    ]
    print_section(f"Top {args.top_n} Overall Recommendations", scored.head(args.top_n), summary_columns)

    for role in ["batter", "bowler", "all_rounder", "wicketkeeper"]:
        role_df = scored.loc[scored["role"] == role].head(3)
        if not role_df.empty:
            print_section(f"Top {role.title()} Recommendations", role_df, summary_columns)

    xi_columns = ["player_name", "role", "runs", "wickets", "final_selector_score"]
    print_section("Recommended T20 Best XI", best_xi, xi_columns)

    value_columns = ["player_name", "role", "final_selector_score", "auction_value_inr_cr", "value_tier"]
    print_section("Top 10 Auction Value Targets", valued.head(10), value_columns)

    venue_columns = ["venue", "player_name", "skill_type", "matches", "venue_score", "venue_rank"]
    print_section("Sample Venue Specialists", venue_scores.head(10), venue_columns)

    matchup_columns = ["batter", "bowler", "balls", "runs", "dismissals", "advantage", "matchup_label"]
    print_section("Sample Batter vs Bowler Matchups", matchups.head(10), matchup_columns)


if __name__ == "__main__":
    main()
