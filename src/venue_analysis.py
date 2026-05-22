from __future__ import annotations

import pandas as pd

from analytics import safe_divide


def build_venue_player_scores(deliveries: pd.DataFrame, min_balls: int = 25, top_n: int = 5) -> pd.DataFrame:
    batting = (
        deliveries.groupby(["venue", "batter"], as_index=False)
        .agg(
            runs=("batter_runs", "sum"),
            balls=("batter_ball_faced", "sum"),
            dismissals=("batter_dismissed", "sum"),
            boundaries=("is_boundary", "sum"),
            matches=("match_id", "nunique"),
        )
    )
    batting = batting.loc[batting["balls"] >= min_balls].copy()
    batting["bat_avg"] = safe_divide(batting["runs"], batting["dismissals"]).where(
        batting["dismissals"] > 0, batting["runs"]
    )
    batting["strike_rate"] = safe_divide(batting["runs"] * 100, batting["balls"])
    batting["venue_batting_score"] = (
        batting["runs"] * 0.20
        + batting["strike_rate"] * 0.45
        + batting["bat_avg"] * 0.25
        + safe_divide(batting["boundaries"] * 100, batting["balls"]) * 0.10
    )
    batting["skill_type"] = "batter"
    batting = batting.rename(columns={"batter": "player_name"})

    bowling = (
        deliveries.groupby(["venue", "bowler"], as_index=False)
        .agg(
            wickets=("wicket_credit", "sum"),
            balls=("legal_ball", "sum"),
            runs_conceded=("bowler_runs_conceded", "sum"),
            dot_balls=("is_dot_ball", "sum"),
            matches=("match_id", "nunique"),
        )
    )
    bowling = bowling.loc[bowling["balls"] >= min_balls].copy()
    bowling["economy"] = safe_divide(bowling["runs_conceded"] * 6, bowling["balls"])
    bowling["dot_pct"] = safe_divide(bowling["dot_balls"] * 100, bowling["balls"])
    bowling["venue_bowling_score"] = (
        bowling["wickets"] * 18
        + (8.5 - bowling["economy"]).clip(lower=0) * 8
        + bowling["dot_pct"] * 0.3
    )
    bowling["skill_type"] = "bowler"
    bowling = bowling.rename(columns={"bowler": "player_name"})

    venue_batting = batting[["venue", "player_name", "skill_type", "matches", "runs", "strike_rate", "venue_batting_score"]]
    venue_batting = venue_batting.rename(columns={"venue_batting_score": "venue_score"})

    venue_bowling = bowling[
        ["venue", "player_name", "skill_type", "matches", "wickets", "economy", "venue_bowling_score"]
    ].rename(columns={"venue_bowling_score": "venue_score"})

    combined = pd.concat([venue_batting, venue_bowling], ignore_index=True, sort=False).fillna(0)
    combined["venue_rank"] = combined.groupby(["venue", "skill_type"])["venue_score"].rank(
        ascending=False, method="dense"
    )
    return combined.loc[combined["venue_rank"] <= top_n].sort_values(
        ["venue", "skill_type", "venue_rank"]
    ).reset_index(drop=True)
