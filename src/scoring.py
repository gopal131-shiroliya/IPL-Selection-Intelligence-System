from __future__ import annotations

import pandas as pd


ROLE_WEIGHTS = {
    "batter": {
        "batting_score": 0.58,
        "bowling_score": 0.02,
        "fielding_score": 0.07,
        "consistency_score": 0.12,
        "recent_form_score": 0.11,
        "team_fit_score": 0.10,
    },
    "bowler": {
        "batting_score": 0.04,
        "bowling_score": 0.56,
        "fielding_score": 0.06,
        "consistency_score": 0.12,
        "recent_form_score": 0.12,
        "team_fit_score": 0.10,
    },
    "all_rounder": {
        "batting_score": 0.28,
        "bowling_score": 0.28,
        "fielding_score": 0.06,
        "consistency_score": 0.12,
        "recent_form_score": 0.16,
        "team_fit_score": 0.10,
    },
    "wicketkeeper": {
        "batting_score": 0.46,
        "bowling_score": 0.00,
        "fielding_score": 0.22,
        "consistency_score": 0.10,
        "recent_form_score": 0.12,
        "team_fit_score": 0.10,
    },
}

SQUAD_TEMPLATE = {
    "batter": 4,
    "bowler": 4,
    "all_rounder": 2,
    "wicketkeeper": 1,
}


def min_max_scale(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    if series.empty or series.max() == series.min():
        return pd.Series([50.0] * len(series), index=series.index)
    return ((series - series.min()) / (series.max() - series.min())) * 100


def inverse_scale(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    if series.empty or series.max() == series.min():
        return pd.Series([50.0] * len(series), index=series.index)
    return ((series.max() - series) / (series.max() - series.min())) * 100


def _team_fit_scores(features: pd.DataFrame) -> pd.Series:
    role_counts = features["role"].value_counts().to_dict()
    scarcity = {}
    for role, demand in SQUAD_TEMPLATE.items():
        available = max(role_counts.get(role, 1), 1)
        scarcity[role] = demand / available
    scarcity_scaled = min_max_scale(pd.Series(scarcity)) * 0.35 + 65
    return features["role"].map(scarcity_scaled.to_dict())


def score_players(features: pd.DataFrame) -> pd.DataFrame:
    scored = features.copy()

    scored["batting_score"] = (
        min_max_scale(scored["runs"]) * 0.18
        + min_max_scale(scored["batting_average"]) * 0.16
        + min_max_scale(scored["strike_rate"]) * 0.22
        + min_max_scale(scored["boundary_percent"]) * 0.10
        + min_max_scale(scored["powerplay_strike_rate"]) * 0.07
        + min_max_scale(scored["middle_overs_strike_rate"]) * 0.07
        + min_max_scale(scored["death_overs_strike_rate"]) * 0.12
        + min_max_scale(scored["thirty_plus_innings"]) * 0.08
    )

    scored["bowling_score"] = (
        min_max_scale(scored["wickets"]) * 0.24
        + inverse_scale(scored["bowling_average"]) * 0.16
        + inverse_scale(scored["economy"]) * 0.22
        + min_max_scale(scored["dot_ball_percent"]) * 0.12
        + inverse_scale(scored["death_overs_economy"]) * 0.12
        + min_max_scale(scored["wickets_per_match"]) * 0.08
        + min_max_scale(scored["maiden_overs"]) * 0.06
    )

    scored["fielding_score"] = (
        min_max_scale(scored["catches"]) * 0.45
        + min_max_scale(scored["stumpings"]) * 0.35
        + min_max_scale(scored["run_outs"]) * 0.20
    )
    scored["consistency_score"] = min_max_scale(scored["consistency_raw"])
    scored["recent_form_score"] = min_max_scale(scored["recent_form_raw"])
    scored["team_fit_score"] = _team_fit_scores(scored)

    final_scores = []
    for _, row in scored.iterrows():
        weights = ROLE_WEIGHTS[row["role"]]
        score = sum(row[column] * weight for column, weight in weights.items())
        final_scores.append(round(score, 2))

    scored["final_selector_score"] = final_scores
    return scored.sort_values("final_selector_score", ascending=False).reset_index(drop=True)


def build_best_xi(scored: pd.DataFrame) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    selected_players: set[str] = set()

    for role, quota in [("wicketkeeper", 1), ("batter", 3), ("all_rounder", 2), ("bowler", 4)]:
        role_candidates = scored.loc[scored["role"] == role]
        for _, row in role_candidates.head(quota).iterrows():
            selected_rows.append(row)
            selected_players.add(row["player_name"])

    remaining = scored.loc[~scored["player_name"].isin(selected_players)].head(1)
    if not remaining.empty:
        selected_rows.append(remaining.iloc[0])

    best_xi = pd.DataFrame(selected_rows).drop_duplicates(subset=["player_name"])
    return best_xi.sort_values("final_selector_score", ascending=False).reset_index(drop=True)
