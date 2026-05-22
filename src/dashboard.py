from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from cricsheet_loader import load_ipl_dataset
from feature_engineering import build_player_features
from matchup_analysis import build_batter_bowler_matchups
from scoring import build_best_xi, score_players
from valuation import estimate_player_value
from venue_analysis import build_venue_player_scores


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"


@st.cache_data(show_spinner=False)
def get_dashboard_data(
    include_current_season: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    deliveries, fielding, matches = load_ipl_dataset(include_incomplete_season=include_current_season)
    available_seasons = sorted(matches["season_year"].unique().tolist())
    selected_seasons = available_seasons[-3:]
    deliveries, fielding, matches = load_ipl_dataset(
        seasons=selected_seasons,
        include_incomplete_season=include_current_season,
    )
    features = build_player_features(deliveries, fielding)
    scored = score_players(features)
    best_xi = build_best_xi(scored)
    valued = estimate_player_value(scored)
    venue_scores = build_venue_player_scores(deliveries)
    matchups = build_batter_bowler_matchups(deliveries)
    OUTPUT_DIR.mkdir(exist_ok=True)
    scored.to_csv(OUTPUT_DIR / "selector_rankings.csv", index=False)
    best_xi.to_csv(OUTPUT_DIR / "best_xi.csv", index=False)
    valued.to_csv(OUTPUT_DIR / "auction_value_rankings.csv", index=False)
    venue_scores.to_csv(OUTPUT_DIR / "venue_specialists.csv", index=False)
    matchups.to_csv(OUTPUT_DIR / "batter_bowler_matchups.csv", index=False)
    return scored, best_xi, matches, valued, venue_scores, matchups


def metric_card_columns(scored: pd.DataFrame, matches: pd.DataFrame) -> None:
    top_player = scored.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Players Ranked", f"{len(scored)}")
    col2.metric("Matches Analysed", f"{matches['match_id'].nunique()}")
    col3.metric("Top Selector Pick", top_player["player_name"])
    col4.metric("Top Score", f"{top_player['final_selector_score']:.2f}")


def build_scatter(scored: pd.DataFrame) -> None:
    fig = px.scatter(
        scored,
        x="strike_rate",
        y="final_selector_score",
        color="role",
        size="matches_played",
        hover_name="player_name",
        hover_data=["runs", "wickets", "economy"],
        title="Selector Score vs Strike Rate",
    )
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)


def build_role_chart(scored: pd.DataFrame) -> None:
    role_summary = (
        scored.groupby("role", as_index=False)["final_selector_score"]
        .mean()
        .sort_values("final_selector_score", ascending=False)
    )
    fig = px.bar(
        role_summary,
        x="role",
        y="final_selector_score",
        color="role",
        title="Average Selector Score by Role",
        text_auto=".2f",
    )
    fig.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def build_best_xi_panel(best_xi: pd.DataFrame) -> None:
    st.subheader("Recommended Best XI")
    st.dataframe(
        best_xi[["player_name", "role", "runs", "wickets", "final_selector_score"]],
        use_container_width=True,
        hide_index=True,
    )


def build_player_compare(scored: pd.DataFrame) -> None:
    st.subheader("Player Comparison")
    options = scored["player_name"].tolist()
    default_players = options[:2] if len(options) >= 2 else options
    selected_players = st.multiselect(
        "Choose players to compare",
        options=options,
        default=default_players,
    )
    if not selected_players:
        return

    compare = scored.loc[scored["player_name"].isin(selected_players)].copy()
    st.dataframe(
        compare[
            [
                "player_name",
                "role",
                "matches_played",
                "runs",
                "wickets",
                "strike_rate",
                "economy",
                "recent_form_score",
                "consistency_score",
                "final_selector_score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def build_value_chart(valued: pd.DataFrame) -> None:
    st.subheader("Auction Value Model")
    fig = px.bar(
        valued.head(12),
        x="player_name",
        y="auction_value_inr_cr",
        color="role",
        title="Top Estimated Auction Values",
        text_auto=".2f",
    )
    fig.update_layout(height=480, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)


def build_venue_specialists(venue_scores: pd.DataFrame) -> None:
    st.subheader("Venue Specialists")
    venues = sorted(venue_scores["venue"].unique().tolist())
    selected_venue = st.selectbox("Choose a venue", venues)
    venue_df = venue_scores.loc[venue_scores["venue"] == selected_venue].copy()
    st.dataframe(venue_df, use_container_width=True, hide_index=True)


def build_matchup_tables(matchups: pd.DataFrame) -> None:
    st.subheader("Batter vs Bowler Matchups")
    left, right = st.columns(2)
    with left:
        st.markdown("**Batter-Favored Matchups**")
        st.dataframe(
            matchups.loc[matchups["matchup_label"] == "batter_favored"].head(10),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.markdown("**Bowler-Favored Matchups**")
        st.dataframe(
            matchups.loc[matchups["matchup_label"] == "bowler_favored"].head(10),
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="IPL Selection Intelligence",
        page_icon="🏏",
        layout="wide",
    )

    st.title("IPL Selection Intelligence Dashboard")
    st.caption("Real ball-by-ball IPL analytics for selectors, scouts, and cricket data science portfolios.")

    include_current = st.sidebar.checkbox("Include current incomplete season", value=False)
    scored, best_xi, matches, valued, venue_scores, matchups = get_dashboard_data(include_current)

    roles = ["All"] + sorted(scored["role"].unique().tolist())
    selected_role = st.sidebar.selectbox("Filter by role", roles)
    min_matches = st.sidebar.slider("Minimum matches played", min_value=8, max_value=50, value=15)
    top_n = st.sidebar.slider("Top players to view", min_value=5, max_value=30, value=10)

    filtered = scored.loc[scored["matches_played"] >= min_matches].copy()
    if selected_role != "All":
        filtered = filtered.loc[filtered["role"] == selected_role].copy()

    metric_card_columns(filtered, matches)

    left, right = st.columns([1.4, 1.0])
    with left:
        build_scatter(filtered.head(60))
    with right:
        build_role_chart(filtered)

    st.subheader("Top Ranked Players")
    st.dataframe(
        filtered.head(top_n)[
            [
                "player_name",
                "role",
                "matches_played",
                "runs",
                "wickets",
                "strike_rate",
                "economy",
                "recent_form_score",
                "consistency_score",
                "final_selector_score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    build_best_xi_panel(best_xi)
    build_player_compare(filtered)
    build_value_chart(valued)
    build_venue_specialists(venue_scores)
    build_matchup_tables(matchups)

    st.subheader("How to Explain This Project")
    st.markdown(
        """
I used real IPL Cricsheet ball-by-ball data to build a role-based player selection engine.
The model engineers T20-specific batting, bowling, fielding, consistency, and recent-form features.
Then it combines those features into a selector score, ranks players by role, and recommends a Best XI.
This makes the project useful for scouting, shortlisting, and data-backed team building discussions.
        """.strip()
    )


if __name__ == "__main__":
    main()
