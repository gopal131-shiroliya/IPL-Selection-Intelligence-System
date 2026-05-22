from __future__ import annotations

import numpy as np
import pandas as pd

from analytics import safe_divide


def _recent_average(frame: pd.DataFrame, player_col: str, value_col: str, window: int = 8) -> pd.Series:
    ordered = frame.sort_values(["match_date", "match_id"])
    recent = ordered.groupby(player_col).tail(window)
    return recent.groupby(player_col)[value_col].mean()


def build_player_features(deliveries: pd.DataFrame, fielding_events: pd.DataFrame) -> pd.DataFrame:
    batting_innings = (
        deliveries.groupby(["batter", "match_id", "match_date"], as_index=False)
        .agg(
            batting_runs=("batter_runs", "sum"),
            balls_faced=("batter_ball_faced", "sum"),
            boundaries=("is_boundary", "sum"),
            dismissed=("batter_dismissed", "sum"),
            powerplay_runs=("batter_runs", lambda s: s[deliveries.loc[s.index, "phase"] == "powerplay"].sum()),
            powerplay_balls=("batter_ball_faced", lambda s: s[deliveries.loc[s.index, "phase"] == "powerplay"].sum()),
            middle_runs=("batter_runs", lambda s: s[deliveries.loc[s.index, "phase"] == "middle"].sum()),
            middle_balls=("batter_ball_faced", lambda s: s[deliveries.loc[s.index, "phase"] == "middle"].sum()),
            death_runs=("batter_runs", lambda s: s[deliveries.loc[s.index, "phase"] == "death"].sum()),
            death_balls=("batter_ball_faced", lambda s: s[deliveries.loc[s.index, "phase"] == "death"].sum()),
        )
    )
    batting_innings = batting_innings.loc[batting_innings["balls_faced"] > 0].copy()
    batting_innings["strike_rate"] = safe_divide(batting_innings["batting_runs"] * 100, batting_innings["balls_faced"])
    batting_innings["batting_impact"] = (
        batting_innings["batting_runs"]
        + np.clip(batting_innings["strike_rate"] - 120, 0, None) * 0.18
        + (batting_innings["boundaries"] * 1.5)
    )

    batting = batting_innings.groupby("batter", as_index=False).agg(
        batting_matches=("match_id", "nunique"),
        innings_batted=("match_id", "count"),
        runs=("batting_runs", "sum"),
        balls_faced=("balls_faced", "sum"),
        dismissals=("dismissed", "sum"),
        boundaries=("boundaries", "sum"),
        powerplay_runs=("powerplay_runs", "sum"),
        powerplay_balls=("powerplay_balls", "sum"),
        middle_runs=("middle_runs", "sum"),
        middle_balls=("middle_balls", "sum"),
        death_runs=("death_runs", "sum"),
        death_balls=("death_balls", "sum"),
        thirty_plus_innings=("batting_runs", lambda s: int((s >= 30).sum())),
        fifty_plus_innings=("batting_runs", lambda s: int((s >= 50).sum())),
        batting_std_dev=("batting_runs", "std"),
    )
    batting["batting_average"] = safe_divide(batting["runs"], batting["dismissals"]).where(
        batting["dismissals"] > 0,
        batting["runs"],
    )
    batting["strike_rate"] = safe_divide(batting["runs"] * 100, batting["balls_faced"])
    batting["boundary_percent"] = safe_divide(batting["boundaries"] * 100, batting["balls_faced"])
    batting["powerplay_strike_rate"] = safe_divide(batting["powerplay_runs"] * 100, batting["powerplay_balls"])
    batting["middle_overs_strike_rate"] = safe_divide(batting["middle_runs"] * 100, batting["middle_balls"])
    batting["death_overs_strike_rate"] = safe_divide(batting["death_runs"] * 100, batting["death_balls"])
    batting["batting_consistency_raw"] = (
        safe_divide(batting["thirty_plus_innings"], batting["innings_batted"]) * 70
        + safe_divide(batting["fifty_plus_innings"], batting["innings_batted"]) * 30
    )
    batting_recent = _recent_average(batting_innings, "batter", "batting_impact").rename("recent_batting_form_raw")
    batting = batting.merge(batting_recent, left_on="batter", right_index=True, how="left")
    batting = batting.rename(columns={"batter": "player_name"})

    bowling_innings = (
        deliveries.groupby(["bowler", "match_id", "match_date"], as_index=False)
        .agg(
            legal_balls=("legal_ball", "sum"),
            runs_conceded=("bowler_runs_conceded", "sum"),
            wickets=("wicket_credit", "sum"),
            dot_balls=("is_dot_ball", "sum"),
            powerplay_runs=("bowler_runs_conceded", lambda s: s[deliveries.loc[s.index, "phase"] == "powerplay"].sum()),
            powerplay_balls=("legal_ball", lambda s: s[deliveries.loc[s.index, "phase"] == "powerplay"].sum()),
            death_runs=("bowler_runs_conceded", lambda s: s[deliveries.loc[s.index, "phase"] == "death"].sum()),
            death_balls=("legal_ball", lambda s: s[deliveries.loc[s.index, "phase"] == "death"].sum()),
        )
    )
    bowling_innings = bowling_innings.loc[bowling_innings["legal_balls"] > 0].copy()
    bowling_innings["economy"] = safe_divide(bowling_innings["runs_conceded"] * 6, bowling_innings["legal_balls"])
    bowling_innings["bowling_impact"] = (
        bowling_innings["wickets"] * 18
        + np.clip(8 - bowling_innings["economy"], 0, None) * 6
        + safe_divide(bowling_innings["dot_balls"] * 100, bowling_innings["legal_balls"]) * 0.1
    )
    bowling_innings["maiden_indicator"] = (
        (bowling_innings["legal_balls"] == 6) & (bowling_innings["runs_conceded"] == 0)
    ).astype(int)

    bowling = bowling_innings.groupby("bowler", as_index=False).agg(
        bowling_matches=("match_id", "nunique"),
        balls_bowled=("legal_balls", "sum"),
        runs_conceded=("runs_conceded", "sum"),
        wickets=("wickets", "sum"),
        dot_balls=("dot_balls", "sum"),
        powerplay_runs_conceded=("powerplay_runs", "sum"),
        powerplay_balls=("powerplay_balls", "sum"),
        death_runs_conceded=("death_runs", "sum"),
        death_balls=("death_balls", "sum"),
        maiden_overs=("maiden_indicator", "sum"),
        bowling_std_dev=("economy", "std"),
    )
    bowling["overs_bowled"] = bowling["balls_bowled"] / 6
    bowling["bowling_average"] = safe_divide(bowling["runs_conceded"], bowling["wickets"]).where(
        bowling["wickets"] > 0,
        bowling["runs_conceded"],
    )
    bowling["economy"] = safe_divide(bowling["runs_conceded"] * 6, bowling["balls_bowled"])
    bowling["dot_ball_percent"] = safe_divide(bowling["dot_balls"] * 100, bowling["balls_bowled"])
    bowling["death_overs_economy"] = safe_divide(bowling["death_runs_conceded"] * 6, bowling["death_balls"])
    bowling["powerplay_economy"] = safe_divide(
        bowling["powerplay_runs_conceded"] * 6, bowling["powerplay_balls"]
    )
    bowling["wickets_per_match"] = safe_divide(bowling["wickets"], bowling["bowling_matches"])
    bowling["bowling_consistency_raw"] = (
        bowling["wickets_per_match"] * 25
        + np.clip(8.5 - bowling["economy"], 0, None) * 12
    )
    bowling_recent = _recent_average(bowling_innings, "bowler", "bowling_impact").rename("recent_bowling_form_raw")
    bowling = bowling.merge(bowling_recent, left_on="bowler", right_index=True, how="left")
    bowling = bowling.rename(columns={"bowler": "player_name"})

    if fielding_events.empty:
        fielding = pd.DataFrame(columns=["player_name", "catches", "stumpings", "run_outs"])
    else:
        fielding = (
            fielding_events.groupby(["fielder", "dismissal_kind"], as_index=False)
            .size()
            .pivot(index="fielder", columns="dismissal_kind", values="size")
            .fillna(0)
            .reset_index()
        )
        fielding.columns.name = None
        fielding = fielding.rename(columns={"fielder": "player_name"})
        fielding["catches"] = fielding.get("caught", 0)
        fielding["stumpings"] = fielding.get("stumped", 0)
        fielding["run_outs"] = fielding.get("run out", 0)
        fielding = fielding[["player_name", "catches", "stumpings", "run_outs"]]

    player_features = batting.merge(bowling, on="player_name", how="outer").merge(fielding, on="player_name", how="left")
    player_features = player_features.fillna(0)
    player_features["matches_played"] = player_features[["batting_matches", "bowling_matches"]].max(axis=1)
    player_features["balls_per_match"] = safe_divide(player_features["balls_faced"], player_features["batting_matches"])
    player_features["overs_per_match"] = safe_divide(player_features["overs_bowled"], player_features["bowling_matches"])

    role = np.select(
        [
            player_features["stumpings"] >= 2,
            (player_features["overs_per_match"] >= 2.0) & (player_features["balls_per_match"] >= 8.0),
            player_features["overs_per_match"] >= 2.0,
        ],
        [
            "wicketkeeper",
            "all_rounder",
            "bowler",
        ],
        default="batter",
    )
    player_features["role"] = role
    player_features["fielding_impact_raw"] = (
        player_features["catches"] * 8 + player_features["stumpings"] * 12 + player_features["run_outs"] * 10
    )
    player_features["recent_form_raw"] = player_features["recent_batting_form_raw"] + player_features["recent_bowling_form_raw"]
    player_features["consistency_raw"] = (
        player_features["batting_consistency_raw"] + player_features["bowling_consistency_raw"]
    )

    player_features = player_features.loc[player_features["matches_played"] >= 8].copy()
    return player_features.sort_values(["matches_played", "runs", "wickets"], ascending=False)
