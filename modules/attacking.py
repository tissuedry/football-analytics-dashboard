
"""Member 1 — Attacking Analytics & Finishing.

Analysis:
- Goals vs Expected Goals (xG)
- Finishing efficiency: Goals - xG
- Shot-Creating Actions (SCA) per 90 minutes

render(df) receives a DataFrame already filtered by app.py.
This module does not read its own CSV and does not mutate the original DataFrame.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.theme import COLORS, COLOR_SEQ, PLOTLY_TEMPLATE


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Get a numeric column; use the default value if the column is missing."""
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safe division to avoid dividing by zero."""
    return numerator.div(denominator.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare attacking metrics without modifying the DataFrame from app.py."""
    data = df.copy().reset_index(drop=True)

    for col in ["goals", "xg", "npxg", "shots", "shots_on_target", "sca", "minutes"]:
        data[col] = _numeric(data, col)

    if "nineties" in data.columns:
        data["nineties"] = _numeric(data, "nineties")
    else:
        data["nineties"] = data["minutes"] / 90.0

    defaults = {
        "player": "Unknown player",
        "team": "—",
        "league": "—",
        "pos": "Other"
    }

    for col, fallback in defaults.items():
        if col not in data.columns:
            data[col] = fallback
        else:
            data[col] = data[col].fillna(fallback).astype(str)

    data["finishing_delta"] = data["goals"] - data["xg"]
    data["goals_per90"] = _safe_divide(data["goals"], data["nineties"])
    data["xg_per90"] = _safe_divide(data["xg"], data["nineties"])
    data["shots_per90"] = _safe_divide(data["shots"], data["nineties"])

    data["shot_accuracy_pct"] = _safe_divide(
        data["shots_on_target"] * 100, data["shots"]
    )

    calculated_sca_per90 = _safe_divide(
        data["sca"], data["nineties"]
    )

    if "sca_per90" in data.columns:
        supplied = pd.to_numeric(data["sca_per90"], errors="coerce")
        data["sca_per90"] = supplied.fillna(calculated_sca_per90)
    else:
        data["sca_per90"] = calculated_sca_per90

    return data


def _player_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "player", "team", "pos", "goals", "xg",
        "finishing_delta", "shots",
        "shot_accuracy_pct", "sca_per90",
    ]

    table = frame[columns].copy().rename(columns={
        "player": "Player",
        "team": "Team",
        "pos": "Position",
        "goals": "Goals",
        "xg": "xG",
        "finishing_delta": "Goals − xG",
        "shots": "Shots",
        "shot_accuracy_pct": "Shot accuracy (%)",
        "sca_per90": "SCA / 90",
    })

    return table.round(2).reset_index(drop=True)


def _goals_vs_xg_chart(data: pd.DataFrame) -> None:
    """Chart comparing actual goals against expected goals."""

    chart_data = data.dropna(subset=["goals", "xg"]).copy()

    if chart_data.empty:
        st.info("Goals and xG data are not available for this filter.")
        return

    position_colors = {
        pos: COLORS.get(pos, COLOR_SEQ[i % len(COLOR_SEQ)])
        for i, pos in enumerate(chart_data["pos"].unique())
    }

    fig = px.scatter(
        chart_data,
        x="xg",
        y="goals",
        color="pos",
        color_discrete_map=position_colors,
        size="shots",
        size_max=19,
        hover_name="player",
        hover_data={
            "team": True,
            "league": True,
            "goals": ":.0f",
            "xg": ":.2f",
            "finishing_delta": ":+.2f",
            "shots": ":.0f",
            "shot_accuracy_pct": ":.1f",
            "sca_per90": ":.2f",
        },
        labels={
            "xg": "Expected Goals (xG)",
            "goals": "Goals scored",
            "pos": "Position"
        },
        title="Actual goals vs expected goals",
    )

    axis_max = max(
        float(chart_data["xg"].max()),
        float(chart_data["goals"].max()),
        1.0
    ) * 1.05

    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=axis_max,
        y1=axis_max,
        line={
            "color": COLORS["muted"],
            "width": 1.5,
            "dash": "dash"
        },
    )

    fig.add_annotation(
        x=axis_max * 0.72,
        y=axis_max * 0.76,
        text="Goals = xG",
        showarrow=False,
        font={"color": COLORS["muted"]},
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=470,
        xaxis_range=[0, axis_max],
        yaxis_range=[0, axis_max],
        legend_title_text="Position",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )


def _finishing_chart(data: pd.DataFrame) -> None:
    """Finishing efficiency chart based on Goals - xG."""

    if data.empty:
        st.info("No players meet the minimum xG threshold.")
        return

    top = data.nlargest(
        10, "finishing_delta"
    ).sort_values("finishing_delta")

    colors = [
        COLORS["positive"] if value >= 0 else COLORS["negative"]
        for value in top["finishing_delta"]
    ]

    fig = go.Figure(go.Bar(
        x=top["finishing_delta"],
        y=top["player"],
        orientation="h",
        marker_color=colors,
        customdata=np.column_stack([
            top["goals"],
            top["xg"]
        ]),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Goals − xG: %{x:+.2f}"
            "<br>Goals: %{customdata[0]:.0f}"
            "<br>xG: %{customdata[1]:.2f}"
            "<extra></extra>"
        ),
    ))

    fig.add_vline(
        x=0,
        line_color=COLORS["muted"],
        line_width=1
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Finishing over/underperformance (Goals − xG)",
        xaxis_title="Goals − xG",
        yaxis_title="",
        height=430,
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )


def _sca_chart(data: pd.DataFrame) -> None:
    """Ranking chart of Shot-Creating Actions per 90 minutes."""

    chart_data = data.dropna(
        subset=["sca_per90"]
    ).copy()

    if chart_data.empty or chart_data["sca_per90"].max() <= 0:
        st.info(
            "Shot-Creating Actions data are not available for this filter."
        )
        return

    top = chart_data.nlargest(
        10, "sca_per90"
    ).sort_values("sca_per90")

    colors = [
        COLORS.get(pos, COLOR_SEQ[0])
        for pos in top["pos"]
    ]

    fig = go.Figure(go.Bar(
        x=top["sca_per90"],
        y=top["player"],
        orientation="h",
        marker_color=colors,
        customdata=np.column_stack([
            top["sca"],
            top["nineties"]
        ]),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "SCA / 90: %{x:.2f}"
            "<br>Total SCA: %{customdata[0]:.0f}"
            "<br>90s played: %{customdata[1]:.1f}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Top Shot-Creating Actions per 90 minutes",
        xaxis_title="SCA per 90 minutes",
        yaxis_title="",
        height=430,
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )


def render(df: pd.DataFrame) -> None:
    """Render the full attacking analysis on the Streamlit dashboard tab."""

    st.subheader("Attacking Analytics & Finishing")

    st.caption(
        "Analysis of goals, chance quality (xG), finishing efficiency, "
        "shot volume, and chance-creation contribution. "
        "Results follow the global filters."
    )

    if df is None or df.empty:
        st.warning(
            "No players match the current filters. "
            "Try lowering the minimum minutes or picking a different league/position."
        )
        return

    if "goals" not in df.columns or "xg" not in df.columns:
        st.error(
            "Required columns 'goals' and/or 'xg' were not found. "
            "Check utils/data_loader.py."
        )
        return

    data = _prepare_data(df)

    total_goals = data["goals"].sum()
    total_xg = data["xg"].sum()
    total_shots = data["shots"].sum()
    total_sot = data["shots_on_target"].sum()
    total_nineties = data["nineties"].sum()
    total_sca = data["sca"].sum()

    shot_accuracy = (
        total_sot / total_shots * 100
        if total_shots > 0 else np.nan
    )

    overall_sca_per90 = (
        total_sca / total_nineties
        if total_nineties > 0 else np.nan
    )

    # KPI utama
    metrics = st.columns(4)

    metrics[0].metric(
        "Goals scored",
        f"{total_goals:,.0f}"
    )

    metrics[1].metric(
        "Expected Goals (xG)",
        f"{total_xg:,.1f}"
    )

    metrics[2].metric(
        "Goals − xG",
        f"{total_goals - total_xg:+.1f}",
        help=(
            "Positive means actual goals exceed xG "
            "expectation; not a standalone proof of finishing skill."
        ),
    )

    metrics[3].metric(
        "Shot accuracy",
        f"{shot_accuracy:.1f}%"
        if pd.notna(shot_accuracy) else "—",
        help="Total shots on target divided by total shots.",
    )

    # 1. GOALS VS EXPECTED GOALS
    st.divider()

    st.markdown("#### 1. Actual Goals vs Expected Goals")

    st.caption(
        "Each point represents one player. Points above the dashed "
        "line scored more goals than xG; points below the line "
        "scored fewer. Point size reflects shot volume."
    )

    _goals_vs_xg_chart(data)

    # 2. FINISHING EFFICIENCY
    st.divider()

    st.markdown("#### 2. Finishing Efficiency")

    max_xg = float(data["xg"].max())

    slider_max = max(
        5.0,
        min(20.0, float(np.ceil(max_xg * 2) / 2))
    )

    min_xg = st.slider(
        "Minimum xG for the finishing ranking",
        min_value=0.0,
        max_value=slider_max,
        value=min(3.0, slider_max),
        step=0.5,
        help=(
            "The xG threshold reduces rankings being skewed "
            "by small samples."
        ),
    )

    eligible = data[data["xg"] >= min_xg].copy()

    st.caption(
        f"{len(eligible)} players meet the xG threshold ≥ {min_xg:.1f}. "
        "Positive values mean overperformance relative to xG."
    )

    finish_col, table_col = st.columns([1.15, 1])

    with finish_col:
        _finishing_chart(eligible)

    with table_col:
        st.markdown("**Players with the highest Goals − xG**")

        if eligible.empty:
            st.info(
                "Lower the minimum xG threshold to see the ranking."
            )
        else:
            st.dataframe(
                _player_table(
                    eligible.nlargest(10, "finishing_delta")
                ),
                use_container_width=True,
                hide_index=True,
            )

    # 3. SHOT-CREATING ACTIONS
    st.divider()

    st.markdown("#### 3. Shot-Creating Actions")

    st.caption(
        "SCA counts offensive actions that lead to a shot. "
        "Per-90 values help compare players with "
        "different playing time."
    )

    sca_col, note_col = st.columns([1.15, 1])

    with sca_col:
        _sca_chart(data)

    with note_col:
        creators = data.dropna(
            subset=["sca_per90"]
        )

        if not creators.empty and creators["sca_per90"].max() > 0:
            creator = creators.loc[
                creators["sca_per90"].idxmax()
            ]

            st.metric(
                "Highest SCA / 90",
                f"{creator['sca_per90']:.2f}"
            )

            st.write(
                f"**{creator['player']}** "
                f"({creator['team']}, {creator['pos']}) "
                f"recorded {creator['sca']:.0f} "
                "Shot-Creating Actions across "
                f"{creator['nineties']:.1f} "
                "90-minute matches played."
            )
        else:
            st.info(
                "No SCA data recorded yet for the selected players."
            )

    # KEY INSIGHT DINAMIS
    insights = []

    if not eligible.empty:
        best = eligible.loc[
            eligible["finishing_delta"].idxmax()
        ]

        worst = eligible.loc[
            eligible["finishing_delta"].idxmin()
        ]

        insights.append(
            f"The player with the highest Goals − xG is "
            f"**{best['player']}** ({best['team']}): "
            f"{best['goals']:.0f} goals from {best['xg']:.1f} xG "
            f"(**{best['finishing_delta']:+.1f}** against expectation)."
        )

        insights.append(
            f"The lowest differential in this sample belongs to "
            f"**{worst['player']}** ({worst['team']}) "
            f"with a Goals − xG of "
            f"**{worst['finishing_delta']:+.1f}**."
        )

    top_scorer = data.loc[
        data["goals"].idxmax()
    ]

    insights.append(
        f"The top scorer under this filter is "
        f"**{top_scorer['player']}** ({top_scorer['team']}) "
        f"with {top_scorer['goals']:.0f} goals "
        f"from {top_scorer['xg']:.1f} xG."
    )

    if pd.notna(overall_sca_per90):
        insights.append(
            f"The weighted average SCA across this sample is "
            f"**{overall_sca_per90:.2f} per 90 minutes**."
        )

    st.divider()

    st.info(
        "**Key Insight**\n\n" + " ".join(insights)
    )

    st.caption(
        "Note: Goals − xG can be affected by variance, penalties, "
        "xG model quality, and sample size. Use it alongside "
        "minutes played, shot volume, and match context."
    )
