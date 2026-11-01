"""EuroScout AI — Football Players Stats 2024-2025: Advanced Scouting &
Performance Intelligence Platform.

Application skeleton: loads data, provides global filters (league, position,
minimum minutes played), then renders each module in its own tab. All
analysis logic lives in modules/*.py — app.py only handles orchestration (Member 5).
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from utils.data_loader import load_data
from utils.theme import COLORS
from modules import overview, attacking, playmaking, defensive, recommender

st.set_page_config(
    page_title="EuroScout AI",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ EuroScout AI")
st.caption("Advanced Scouting & Performance Intelligence Platform — Europe's Big 5 Leagues 2024-2025")

df = load_data()

with st.sidebar:
    st.header("Filters")

    leagues = sorted(df["league"].dropna().unique().tolist())
    selected_leagues = st.multiselect("League", leagues, default=leagues)

    positions = ["FW", "MF", "DF", "GK"]
    selected_positions = st.multiselect("Position", positions, default=positions)

    min_minutes = st.slider(
        "Minimum minutes played",
        min_value=0,
        max_value=int(df["minutes"].max()),
        value=900,
        step=90,
        help="Players below this minutes threshold are excluded to avoid biased stats.",
    )

filtered_df = df[
    df["league"].isin(selected_leagues)
    & df["pos"].isin(selected_positions)
    & (df["minutes"] >= min_minutes)
]

st.caption(f"{len(filtered_df)} players match the current filters.")

tab_overview, tab_attacking, tab_playmaking, tab_defensive, tab_recommender = st.tabs(
    [
        "🌐 League Overview",
        "⚡ Attacking & Finishing",
        "🎯 Passing & Playmaking",
        "🛡️ Defensive Workrate",
        "🔍 ML Scouting & Similarity",
    ]
)

with tab_overview:
    overview.render(filtered_df)

with tab_attacking:
    attacking.render(filtered_df)

with tab_playmaking:
    playmaking.render(filtered_df)

with tab_defensive:
    defensive.render(filtered_df)

with tab_recommender:
    recommender.render(filtered_df)
