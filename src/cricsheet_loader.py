from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ipl_json"


def season_to_year(season_value: object) -> int:
    season_text = str(season_value)
    if "/" in season_text:
        season_text = season_text.split("/")[0]
    return int(season_text)


def phase_for_over(over_number: int) -> str:
    if over_number < 6:
        return "powerplay"
    if over_number < 15:
        return "middle"
    return "death"


def _normalise_fielders(raw_fielders: Iterable[object]) -> list[str]:
    names: list[str] = []
    for fielder in raw_fielders:
        if isinstance(fielder, dict):
            names.append(str(fielder.get("name")))
        else:
            names.append(str(fielder))
    return names


def load_ipl_dataset(
    data_dir: Path | None = None,
    seasons: list[int] | None = None,
    include_incomplete_season: bool = False,
    max_matches: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = data_dir or DATA_DIR
    match_files = sorted(raw_dir.glob("*.json"))
    if max_matches is not None:
        match_files = match_files[:max_matches]

    deliveries: list[dict[str, object]] = []
    fielding_events: list[dict[str, object]] = []
    matches: list[dict[str, object]] = []

    for match_file in match_files:
        match = json.loads(match_file.read_text(encoding="utf-8"))
        info = match["info"]
        match_dates = [date.fromisoformat(value) for value in info.get("dates", [])]
        match_date = max(match_dates)
        season = str(info.get("season"))
        season_year = season_to_year(season)
        teams = list(info.get("teams", []))

        matches.append(
            {
                "match_id": match_file.stem,
                "match_date": match_date,
                "season": season,
                "season_year": season_year,
                "venue": info.get("venue", "Unknown"),
                "city": info.get("city", "Unknown"),
                "team_1": teams[0] if len(teams) > 0 else "Unknown",
                "team_2": teams[1] if len(teams) > 1 else "Unknown",
            }
        )

        for innings_index, innings in enumerate(match["innings"], start=1):
            batting_team = innings["team"]
            bowling_team = next((team for team in teams if team != batting_team), "Unknown")

            for over in innings["overs"]:
                over_number = int(over["over"])
                phase = phase_for_over(over_number)

                for ball_index, delivery in enumerate(over["deliveries"], start=1):
                    runs = delivery["runs"]
                    extras = delivery.get("extras", {})
                    is_wide = "wides" in extras
                    is_no_ball = "noballs" in extras
                    legal_ball = 0 if (is_wide or is_no_ball) else 1
                    batter_ball_faced = 0 if is_wide else 1
                    wickets = delivery.get("wickets", [])

                    wicket_credit = 0
                    batter_dismissed = 0
                    for wicket in wickets:
                        dismissal_kind = wicket["kind"]
                        player_out = wicket["player_out"]
                        if (
                            dismissal_kind
                            not in {"run out", "retired hurt", "retired out", "obstructing the field"}
                        ):
                            wicket_credit += 1
                        if player_out == delivery["batter"] and dismissal_kind not in {"retired hurt", "retired out"}:
                            batter_dismissed += 1

                        for fielder_name in _normalise_fielders(wicket.get("fielders", [])):
                            fielding_events.append(
                                {
                                    "match_id": match_file.stem,
                                    "match_date": match_date,
                                    "season": season,
                                    "season_year": season_year,
                                    "fielder": fielder_name,
                                    "dismissal_kind": dismissal_kind,
                                }
                            )

                    deliveries.append(
                        {
                            "match_id": match_file.stem,
                            "match_date": match_date,
                            "season": season,
                            "season_year": season_year,
                            "venue": info.get("venue", "Unknown"),
                            "city": info.get("city", "Unknown"),
                            "innings": innings_index,
                            "batting_team": batting_team,
                            "bowling_team": bowling_team,
                            "over": over_number,
                            "ball_in_over": ball_index,
                            "phase": phase,
                            "batter": delivery["batter"],
                            "bowler": delivery["bowler"],
                            "non_striker": delivery["non_striker"],
                            "batter_runs": runs["batter"],
                            "extras": runs["extras"],
                            "total_runs": runs["total"],
                            "bowler_runs_conceded": runs["total"] - extras.get("byes", 0) - extras.get("legbyes", 0),
                            "is_wide": int(is_wide),
                            "is_no_ball": int(is_no_ball),
                            "legal_ball": legal_ball,
                            "batter_ball_faced": batter_ball_faced,
                            "is_boundary": int(runs["batter"] in {4, 6}),
                            "is_dot_ball": int(legal_ball == 1 and runs["total"] == 0),
                            "wicket_credit": wicket_credit,
                            "batter_dismissed": batter_dismissed,
                        }
                    )

    deliveries_df = pd.DataFrame(deliveries)
    fielding_df = pd.DataFrame(fielding_events)
    matches_df = pd.DataFrame(matches)

    if deliveries_df.empty:
        raise ValueError(f"No IPL match files found in {raw_dir}")

    deliveries_df["match_date"] = pd.to_datetime(deliveries_df["match_date"])
    matches_df["match_date"] = pd.to_datetime(matches_df["match_date"])
    if not fielding_df.empty:
        fielding_df["match_date"] = pd.to_datetime(fielding_df["match_date"])

    if seasons:
        deliveries_df = deliveries_df.loc[deliveries_df["season_year"].isin(seasons)].copy()
        matches_df = matches_df.loc[matches_df["season_year"].isin(seasons)].copy()
        if not fielding_df.empty:
            fielding_df = fielding_df.loc[fielding_df["season_year"].isin(seasons)].copy()
    elif not include_incomplete_season:
        latest_season_year = int(matches_df["season_year"].max())
        deliveries_df = deliveries_df.loc[deliveries_df["season_year"] != latest_season_year].copy()
        matches_df = matches_df.loc[matches_df["season_year"] != latest_season_year].copy()
        if not fielding_df.empty:
            fielding_df = fielding_df.loc[fielding_df["season_year"] != latest_season_year].copy()

    return deliveries_df, fielding_df, matches_df
