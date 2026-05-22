from __future__ import annotations

import pandas as pd

from scoring import min_max_scale


ROLE_BASE_PRICE = {
    "batter": 4.5,
    "bowler": 4.2,
    "all_rounder": 6.0,
    "wicketkeeper": 4.8,
}


def estimate_player_value(scored: pd.DataFrame) -> pd.DataFrame:
    valued = scored.copy()
    valued["impact_index"] = (
        valued["batting_score"] * 0.35
        + valued["bowling_score"] * 0.30
        + valued["recent_form_score"] * 0.15
        + valued["consistency_score"] * 0.10
        + valued["fielding_score"] * 0.10
    )
    valued["brand_index"] = (
        min_max_scale(valued["matches_played"]) * 0.40
        + min_max_scale(valued["runs"] + (valued["wickets"] * 22)) * 0.60
    )
    valued["auction_value_inr_cr"] = (
        valued["role"].map(ROLE_BASE_PRICE)
        + (valued["final_selector_score"] / 100) * 7.5
        + (valued["impact_index"] / 100) * 5.0
        + (valued["brand_index"] / 100) * 3.0
    ).round(2)

    valued["value_tier"] = pd.cut(
        valued["auction_value_inr_cr"],
        bins=[0, 4, 7, 10, 14, 25],
        labels=["budget", "solid", "premium", "elite", "marquee"],
        include_lowest=True,
    )
    return valued.sort_values("auction_value_inr_cr", ascending=False).reset_index(drop=True)
