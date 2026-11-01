"""Passing & Playmaking analytics for EuroScout AI."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.theme import COLORS, COLOR_SEQ, PLOTLY_TEMPLATE

IDENTITY = ["player", "team", "league", "pos", "minutes"]

METRICS = {
    "key_passes": "Key Passes",
    "prog_passes": "Progressive Passes",
    "xag": "Expected Assisted Goals (xAG)",
    "assists": "Assists",
    "sca": "Shot-Creating Actions (SCA)",
    "gca": "Goal-Creating Actions (GCA)",
    "prog_carries": "Progressive Carries",
}

SMALL_SAMPLE_MINUTES = 900


def _prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate and prepare a local copy of the globally filtered data."""
    missing = [c for c in IDENTITY if c not in df.columns]
    if missing:
        raise ValueError("Required columns not found: " + ", ".join(missing))

    data = df.copy(deep=True).reset_index(drop=True)
    data["record_id"] = np.arange(len(data)) + 1
    notes = []

    for col in IDENTITY[:-1]:
        data[col] = data[col].astype("string").str.strip().replace("", pd.NA)
        data[col] = data[col].fillna("Unknown")

    data["pos"] = data["pos"].str.split(",").str[0].str.strip().str.upper()

    # Drop multi-team total rows to avoid double counting.
    total_rows = data["team"].str.fullmatch(
        r"(?i)(?:\d+\s+(?:squads?|clubs?|teams?)|(?:all\s+)?(?:squads|clubs|teams)|total)",
        na=False,
    )
    if total_rows.any():
        notes.append(f"{int(total_rows.sum())} multi-team total rows excluded to avoid double counting.")
        data = data.loc[~total_rows].copy()

    numeric_cols = [
        "minutes", "key_passes", "prog_passes", "xag", "assists",
        "sca", "gca", "prog_carries", "cmp", "att", "cmp_pct",
    ]
    for col in numeric_cols:
        if col not in data.columns:
            data[col] = np.nan
            if col in METRICS:
                notes.append(f"Column {col} is not available; the related metric cannot be analyzed.")
            continue

        original = data[col]
        values = pd.to_numeric(original, errors="coerce").astype(float)
        valid = np.isfinite(values) & (values >= 0)
        if col == "minutes":
            valid &= values > 0
        if col == "cmp_pct":
            valid &= values <= 100
        invalid = original.notna() & ~valid
        if invalid.any():
            notes.append(f"{col}: {int(invalid.sum())} invalid values treated as missing.")
        data[col] = values.where(valid)

    # Compute completion percentage only when it isn't already available.
    calculated_pct = data["cmp"] / data["att"].where(data["att"] > 0) * 100
    data["pass_completion_pct"] = data["cmp_pct"].where(
        data["cmp_pct"].notna(), calculated_pct
    )

    # Per-90-minute rates. Missing values are not converted to zero.
    for metric in METRICS:
        data[metric + "_per90"] = (
            data[metric] * 90 / data["minutes"]
        ).replace([np.inf, -np.inf], np.nan)

    data["label"] = (
        data["player"].astype(str) + " · " +
        data["team"].astype(str) + " · " +
        data["league"].astype(str) + " [#" +
        data["record_id"].astype(str) + "]"
    )
    return data, notes


def _show_chart(fig, key: str, height: int = 440):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo": False})


def _ranking_chart(data: pd.DataFrame, metric: str, top_n: int, mode: str):
    column = metric + "_per90" if mode == "Per 90 Minutes" else metric
    ranked = data.dropna(subset=[column]).sort_values(
        [column, "player", "team", "league", "record_id"],
        ascending=[False, True, True, True, True],
        kind="stable",
    ).head(top_n)

    if ranked.empty:
        st.info("No data available for this metric and filter.")
        return

    unit = "per 90 minutes" if mode == "Per 90 Minutes" else "total"
    fig = px.bar(
        ranked.sort_values(column),
        x=column,
        y="label",
        orientation="h",
        text=column,
        color="pos",
        color_discrete_map={k: COLORS.get(k, COLORS["muted"]) for k in ["FW", "MF", "DF", "GK"]},
        hover_data={"team": True, "league": True, "minutes": ":.0f", "pos": True},
        labels={column: f"{METRICS[metric]} ({unit})", "label": "Player", "pos": "Position"},
        title=f"Top {len(ranked)} Players — {METRICS[metric]}",
    )
    fig.update_traces(texttemplate="%{x:.2f}" if mode == "Per 90 Minutes" else "%{x:.0f}", textposition="outside")
    _show_chart(fig, f"pm_rank_{metric}_{mode}")


def render(df: pd.DataFrame):
    """Render the complete Passing & Playmaking tab."""
    st.subheader("Passing & Playmaking")
    st.caption(
        "Analysis of creativity, ball distribution, and player contribution to chance creation. "
        "All analysis follows the league, position, and minutes played filters in the sidebar."
    )

    if df is None or df.empty:
        st.info("No players match the global filters. Please adjust the filters in the sidebar.")
        return

    try:
        data, notes = _prepare_data(df)
    except ValueError as exc:
        st.error(str(exc))
        return

    if data.empty:
        st.info("No player data available to analyze.")
        return

    valid_minutes = data["minutes"].dropna()
    if notes:
        with st.expander("Data validation and availability"):
            for note in notes:
                st.write("• " + note)

    # ---------------- KPI ----------------
    st.markdown("#### Playmaking Summary")
    total_kp = data["key_passes"].sum(min_count=1)
    total_prog = data["prog_passes"].sum(min_count=1)
    total_xag = data["xag"].sum(min_count=1)
    total_assists = data["assists"].sum(min_count=1)
    total_sca = data["sca"].sum(min_count=1)

    kpis = st.columns(5)
    kpis[0].metric("Total Key Passes", f"{total_kp:,.0f}" if pd.notna(total_kp) else "—")
    kpis[1].metric("Progressive Passes", f"{total_prog:,.0f}" if pd.notna(total_prog) else "—")
    kpis[2].metric("Total xAG", f"{total_xag:,.2f}" if pd.notna(total_xag) else "—")
    kpis[3].metric("Total Assists", f"{total_assists:,.0f}" if pd.notna(total_assists) else "—")
    kpis[4].metric("Total SCA", f"{total_sca:,.0f}" if pd.notna(total_sca) else "—")

    st.caption(
        f"{len(data):,} player–team records analyzed · "
        f"{data['player'].nunique():,} player names · "
        f"{valid_minutes.sum():,.0f} total valid minutes."
    )

    # ---------------- Ranking ----------------
    st.divider()
    st.markdown("#### 1. Playmaking Leaderboard")
    st.caption(
        "Pick a metric to see the players with the highest playmaking contribution. "
        "Per 90 minutes helps compare players with different playing time."
    )
    controls = st.columns(3)
    selected_metric = controls[0].selectbox(
        "Ranking metric",
        list(METRICS.keys()),
        format_func=METRICS.get,
        key="pm_metric",
    )
    ranking_mode = controls[1].radio(
        "Ranking unit", ["Per 90 Minutes", "Total"], horizontal=True, key="pm_mode"
    )
    top_n = controls[2].selectbox("Number of players", [5, 10, 15, 20], index=1, key="pm_top_n")
    _ranking_chart(data, selected_metric, top_n, ranking_mode)

    # ---------------- Creative profile ----------------
    st.divider()
    st.markdown("#### 2. Top Chance Creators")
    st.caption(
        "Chart comparing Key Passes, xAG, and Assists. "
        "Use per-90-minutes mode to reduce the effect of playing time differences."
    )
    creative_mode = st.radio(
        "Creativity chart unit",
        ["Per 90 Minutes", "Total"],
        horizontal=True,
        key="pm_creative_mode",
    )
    creative_metrics = ["key_passes", "xag", "assists"]
    value_columns = [
        m + "_per90" if creative_mode == "Per 90 Minutes" else m
        for m in creative_metrics
    ]
    creative = data[["label", "player", "team", "pos", "minutes"] + value_columns].copy()
    creative = creative.rename(columns={
        value_columns[0]: "Key Passes",
        value_columns[1]: "xAG",
        value_columns[2]: "Assists",
    })
    creative_long = creative.melt(
        id_vars=["label", "player", "team", "pos", "minutes"],
        value_vars=["Key Passes", "xAG", "Assists"],
        var_name="Metric",
        value_name="Value",
    ).dropna(subset=["Value"])

    if creative_long.empty:
        st.info("Creativity data is not available yet.")
    else:
        # Take top players by combined metric ranking so the chart stays readable.
        top_players = (
            creative.groupby("label")[["Key Passes", "xAG", "Assists"]]
            .sum(min_count=1).fillna(0).sum(axis=1)
            .nlargest(top_n).index
        )
        creative_long = creative_long[creative_long["label"].isin(top_players)]
        fig = px.bar(
            creative_long,
            x="player",
            y="Value",
            color="Metric",
            barmode="group",
            hover_data=["team", "pos", "minutes"],
            color_discrete_sequence=COLOR_SEQ,
            labels={"player": "Player", "Value": "Value", "Metric": "Metric"},
            title=f"Player Creativity Comparison ({creative_mode})",
        )
        _show_chart(fig, "pm_creative_chart", 480)

    # ---------------- Key passes vs xAG ----------------
    st.divider()
    st.markdown("#### 3. Key Passes vs Expected Assisted Goals Relationship")
    st.caption(
        "Each point represents one player. Players to the right have more key passes; "
        "players higher up have a higher xAG. Correlation shows a relationship, not causation."
    )
    scatter = data.dropna(subset=["key_passes_per90", "xag_per90"]).copy()
    if len(scatter) >= 2:
        fig = px.scatter(
            scatter,
            x="key_passes_per90",
            y="xag_per90",
            color="pos",
            color_discrete_map={k: COLORS.get(k, COLORS["muted"]) for k in ["FW", "MF", "DF", "GK"]},
            size="sca",
            size_max=22,
            hover_name="player",
            hover_data={"team": True, "league": True, "minutes": ":.0f", "sca": True},
            labels={
                "key_passes_per90": "Key Passes per 90",
                "xag_per90": "xAG per 90",
                "pos": "Position",
                "sca": "SCA",
            },
            title="Key Passes per 90 vs xAG per 90",
        )
        _show_chart(fig, "pm_keypasses_xag", 480)
        corr = scatter["key_passes_per90"].corr(scatter["xag_per90"])
        st.metric("Pearson Correlation: Key Passes and xAG", f"{corr:.3f}" if pd.notna(corr) else "—")
    else:
        st.info("At least two players with Key Passes and xAG data are needed to build a scatter plot.")

    # ---------------- Passing accuracy ----------------
    st.divider()
    st.markdown("#### 4. Passing Accuracy")
    st.caption(
        "Pass completion percentage is computed from completed passes divided by total pass attempts "
        "when the source percentage is not available."
    )
    passing = data.dropna(subset=["pass_completion_pct"]).copy()
    passing = passing.sort_values(
        ["pass_completion_pct", "player"], ascending=[False, True]
    ).head(top_n)
    if passing.empty:
        st.info("Passing completion data is not available.")
    else:
        fig = px.bar(
            passing.sort_values("pass_completion_pct"),
            x="pass_completion_pct",
            y="label",
            orientation="h",
            text="pass_completion_pct",
            color="pos",
            color_discrete_map={k: COLORS.get(k, COLORS["muted"]) for k in ["FW", "MF", "DF", "GK"]},
            hover_data={"cmp": ":.0f", "att": ":.0f", "minutes": ":.0f"},
            labels={"pass_completion_pct": "Pass Completion (%)", "label": "Player", "pos": "Position"},
            title=f"Top {len(passing)} Players by Pass Accuracy",
        )
        fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
        _show_chart(fig, "pm_passing_accuracy")

    # ---------------- Player table ----------------
    st.divider()
    st.markdown("#### 5. Player Playmaking Statistics")
    search = st.text_input("Search player or team name", key="pm_search").strip().lower()
    table = data.copy()
    if search:
        mask = (
            table["player"].str.lower().str.contains(search, na=False)
            | table["team"].str.lower().str.contains(search, na=False)
        )
        table = table[mask]

    display_cols = [
        "player", "team", "league", "pos", "minutes",
        "key_passes", "key_passes_per90",
        "prog_passes", "prog_passes_per90",
        "xag", "xag_per90", "assists", "sca", "gca",
        "prog_carries", "pass_completion_pct",
    ]
    display_cols = [c for c in display_cols if c in table.columns]
    rename = {
        "player": "Player", "team": "Team", "league": "League", "pos": "Position",
        "minutes": "Minutes", "key_passes": "Key Passes",
        "key_passes_per90": "KP / 90", "prog_passes": "Progressive Passes",
        "prog_passes_per90": "Prog. Passes / 90", "xag": "xAG",
        "xag_per90": "xAG / 90", "assists": "Assists", "sca": "SCA",
        "gca": "GCA", "prog_carries": "Progressive Carries",
        "pass_completion_pct": "Pass Completion (%)",
    }
    st.dataframe(
        table[display_cols].rename(columns=rename).round(2),
        use_container_width=True,
        hide_index=True,
    )

    # ---------------- Key insight ----------------
    st.divider()
    st.markdown("#### 6. Key Insight")
    insights = []
    for metric, title in [
        ("key_passes_per90", "Key Passes per 90"),
        ("prog_passes_per90", "Progressive Passes per 90"),
        ("xag_per90", "xAG per 90"),
        ("sca_per90", "SCA per 90"),
    ]:
        if metric in data and data[metric].notna().any():
            best = data.loc[data[metric].idxmax()]
            insights.append(
                f"- **{title}:** {best['player']} ({best['team']}) recorded "
                f"{best[metric]:.2f}."
            )

    if insights:
        st.markdown("\n".join(insights))
    else:
        st.info("Not enough data yet to generate automatic insights.")

    below_threshold = int(data["minutes"].lt(SMALL_SAMPLE_MINUTES).sum())
    st.caption(
        f"Interpretation note: {below_threshold} of {len(data)} records have playing time "
        f"below {SMALL_SAMPLE_MINUTES} minutes. Per-90 stats from small minute samples "
        "can vary significantly. Creativity numbers measure volume/involvement, "
        "not overall decision-making quality."
    )
