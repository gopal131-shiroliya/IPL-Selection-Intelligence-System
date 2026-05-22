from __future__ import annotations

import pandas as pd

from analytics import safe_divide


def build_batter_bowler_matchups(
    deliveries: pd.DataFrame,
    min_balls: int = 12,
    top_n: int = 25,
) -> pd.DataFrame:
    matchups = (
        deliveries.groupby(["batter", "bowler"], as_index=False)
        .agg(
            balls=("batter_ball_faced", "sum"),
            runs=("batter_runs", "sum"),
            dismissals=("batter_dismissed", "sum"),
            boundaries=("is_boundary", "sum"),
            matches=("match_id", "nunique"),
        )
    )
    matchups = matchups.loc[matchups["balls"] >= min_balls].copy()
    matchups["strike_rate"] = safe_divide(matchups["runs"] * 100, matchups["balls"])
    matchups["boundary_percent"] = safe_divide(matchups["boundaries"] * 100, matchups["balls"])
    matchups["dismissal_rate"] = safe_divide(matchups["dismissals"] * 100, matchups["balls"])
    matchups["batter_advantage_score"] = (
        matchups["strike_rate"] * 0.55
        + matchups["boundary_percent"] * 0.25
        - matchups["dismissal_rate"] * 1.20
        + matchups["matches"] * 1.5
    )
    matchups["bowler_advantage_score"] = (
        matchups["dismissal_rate"] * 1.8
        + (130 - matchups["strike_rate"]).clip(lower=0) * 0.45
        + (25 - matchups["boundary_percent"]).clip(lower=0) * 0.60
    )
    matchups["advantage"] = matchups["batter_advantage_score"] - matchups["bowler_advantage_score"]
    matchups = matchups.sort_values("advantage", ascending=False).reset_index(drop=True)
    top_batter = matchups.head(top_n).copy()
    top_batter["matchup_label"] = "batter_favored"
    top_bowler = matchups.sort_values("advantage", ascending=True).head(top_n).copy()
    top_bowler["matchup_label"] = "bowler_favored"
    return pd.concat([top_batter, top_bowler], ignore_index=True)
