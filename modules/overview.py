"""Member 5 — Executive Overview & League Intelligence.

Shows a macro summary of competition performance, age demographics,
cross-league productivity comparison, and top-player spotlights across
categories in Europe's Big 5 leagues.
Contract: a single render(df) function that fills this entire tab.
Use PLOTLY_TEMPLATE, COLORS, COLOR_SEQ from utils/theme.py for all charts.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'utils' can always be imported
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.theme import COLORS, COLOR_SEQ, PLOTLY_TEMPLATE


def _league_productivity_chart(df: pd.DataFrame) -> None:
    """Chart comparing actual goals productivity vs xG per league."""
    if "league" not in df.columns or "goals" not in df.columns or "xg" not in df.columns:
        st.info("League, goals, or xG data is incomplete.")
        return

    league_stats = (
        df.groupby("league")
        .agg(
            total_goals=("goals", "sum"),
            total_xg=("xg", "sum"),
            total_players=("player", "count"),
        )
        .reset_index()
    )

    if league_stats.empty:
        st.info("No data available for the league chart.")
        return

    league_stats["finishing_diff"] = league_stats["total_goals"] - league_stats["total_xg"]
    league_stats = league_stats.sort_values("total_goals", ascending=False)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=league_stats["league"],
            y=league_stats["total_goals"],
            name="Goals Scored",
            marker_color=COLORS["primary"],
            hovertemplate="<b>%{x}</b><br>Goals: %{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=league_stats["league"],
            y=league_stats["total_xg"],
            name="Expected Goals (xG)",
            marker_color=COLORS["accent"],
            hovertemplate="<b>%{x}</b><br>xG: %{y:.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="group",
        title="Total Goals Scored vs Expected Goals (xG) by League",
        xaxis_title="",
        yaxis_title="Total Output",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _age_distribution_chart(df: pd.DataFrame) -> None:
    """Age distribution of players by position."""
    valid_age = df.dropna(subset=["age"]).copy()
    if valid_age.empty:
        st.info("Age data is not available.")
        return

    pos_colors = {
        pos: COLORS.get(pos, COLOR_SEQ[i % len(COLOR_SEQ)])
        for i, pos in enumerate(sorted(valid_age["pos"].unique()))
    }

    fig = px.box(
        valid_age,
        x="pos",
        y="age",
        color="pos",
        color_discrete_map=pos_colors,
        points="outliers",
        labels={"pos": "Position", "age": "Age (Years)"},
        title="Age Profile & Distribution by Position Group",
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=420,
        showlegend=False,
        yaxis=dict(gridcolor=COLORS["grid"]),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _team_goals_chart(df: pd.DataFrame) -> None:
    """Top 10 clubs most prolific at scoring goals."""
    if "team" not in df.columns or "goals" not in df.columns:
        return

    team_stats = (
        df.groupby(["team", "league"])
        .agg(total_goals=("goals", "sum"), total_xg=("xg", "sum"))
        .reset_index()
        .sort_values("total_goals", ascending=True)
        .tail(10)
    )

    if team_stats.empty:
        return

    fig = go.Figure(
        go.Bar(
            x=team_stats["total_goals"],
            y=team_stats["team"],
            orientation="h",
            marker_color=COLORS["primary"],
            customdata=np.column_stack([team_stats["league"], team_stats["total_xg"]]),
            hovertemplate=(
                "<b>%{y}</b> (%{customdata[0]})<br>"
                "Goals: %{x:.0f}<br>"
                "xG: %{customdata[1]:.1f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Top 10 Most Prolific Teams (Goals Scored)",
        xaxis_title="Goals Scored",
        yaxis_title="",
        height=380,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def render(df: pd.DataFrame) -> None:
    """Render all visualizations and macro analysis on the Overview tab."""
    st.subheader("Executive Overview & League Intelligence")
    st.caption(
        "Macro-level perspective across Europe's Big 5 Leagues (2024–2025). "
        "Provides league-wide benchmarks, squad demographic profiles, and multi-category standouts."
    )

    if df is None or df.empty:
        st.warning("No players match the current filters. Please adjust the filters in the sidebar.")
        return

    # Top-level KPIs
    total_players = len(df)
    total_goals = float(df["goals"].sum()) if "goals" in df.columns else 0.0
    total_xg = float(df["xg"].sum()) if "xg" in df.columns else 0.0
    avg_age = float(df["age"].mean()) if "age" in df.columns else 0.0
    total_matches = float(df["nineties"].sum()) if "nineties" in df.columns else (
        float(df["minutes"].sum()) / 90.0 if "minutes" in df.columns else 0.0
    )

    cols = st.columns(4)
    cols[0].metric("Filtered Players", f"{total_players:,}")
    cols[1].metric("Total Goals", f"{total_goals:,.0f}")
    cols[2].metric("Total Expected Goals", f"{total_xg:,.1f}", delta=f"{total_goals - total_xg:+.1f} Net")
    cols[3].metric("Average Age", f"{avg_age:.1f} yrs" if avg_age > 0 else "—")

    st.divider()

    # 1. League Output & Age Profile side-by-side
    col_left, col_right = st.columns(2)
    with col_left:
        _league_productivity_chart(df)
    with col_right:
        _age_distribution_chart(df)

    st.divider()

    # 2. Team Productivity & Top Leaders
    col_team, col_leaders = st.columns([1.1, 1])

    with col_team:
        _team_goals_chart(df)

    with col_leaders:
        st.markdown("#### Category Leaders Spotlight")

        top_scorer = df.loc[df["goals"].idxmax()] if ("goals" in df.columns and not df.empty) else None
        top_assister = df.loc[df["assists"].idxmax()] if ("assists" in df.columns and not df.empty) else None
        top_tackler = df.loc[df["tackles"].idxmax()] if ("tackles" in df.columns and not df.empty) else None

        if top_scorer is not None:
            st.info(
                f"⚽ **Top Scorer:** **{top_scorer['player']}** ({top_scorer.get('team', '—')}) — "
                f"**{top_scorer['goals']:.0f} Goals** ({top_scorer.get('xg', 0):.1f} xG)"
            )

        if top_assister is not None:
            st.success(
                f"🎯 **Top Playmaker:** **{top_assister['player']}** ({top_assister.get('team', '—')}) — "
                f"**{top_assister['assists']:.0f} Assists** ({top_assister.get('xag', 0):.1f} xAG)"
            )

        if top_tackler is not None:
            st.warning(
                f"🛡️ **Top Ball-Winner:** **{top_tackler['player']}** ({top_tackler.get('team', '—')}) — "
                f"**{top_tackler['tackles']:.0f} Tackles**"
            )

    # 3. Dynamic Key Insight
    st.divider()
    insights = []
    if "league" in df.columns and "goals" in df.columns:
        league_sum = df.groupby("league")["goals"].sum()
        if not league_sum.empty:
            top_league = league_sum.idxmax()
            top_league_goals = league_sum.max()
            insights.append(
                f"The league with the most goals scored is **{top_league}** with a total of **{top_league_goals:,.0f} goals**."
            )

    if total_matches > 0 and total_goals > 0:
        gpg = total_goals / total_matches
        insights.append(
            f"The average goal output in this sample is **{gpg:.2f} goals per 90 minutes** played."
        )

    if avg_age > 0:
        insights.append(
            f"The average age of players tracked here is **{avg_age:.1f} years**, with most players in their prime age range (23–29 years)."
        )

    st.info("**Key Insight**\n\n" + " ".join(insights))


if __name__ == "__main__":
    from utils.data_loader import load_data
    st.set_page_config(page_title="Executive Overview", page_icon="🌐", layout="wide")
    st.title("⚽ EuroScout AI — League Overview")
    _df = load_data()
    render(_df)


