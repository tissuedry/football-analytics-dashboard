"""Shared visual theme for all Plotly charts in EuroScout AI.

All modules (attacking, playmaking, defensive, recommender) MUST use
PLOTLY_TEMPLATE, COLORS, and COLOR_SEQ from this file so the dashboard
looks consistent.
"""

import plotly.graph_objects as go
import plotly.io as pio

COLORS = {
    "background": "#0E1117",
    "surface": "#161B22",
    "text": "#E6EDF3",
    "muted": "#8B949E",
    "grid": "#30363D",
    "primary": "#2DD4BF",
    "accent": "#F59E0B",
    "positive": "#4ADE80",
    "negative": "#F87171",
    # fixed colors per position, used across all modules
    "FW": "#F87171",
    "MF": "#2DD4BF",
    "DF": "#60A5FA",
    "GK": "#FBBF24",
}

# categorical color sequence for multi-series charts (leagues, teams, players, etc.)
COLOR_SEQ = [
    "#2DD4BF", "#F59E0B", "#60A5FA", "#F87171",
    "#A78BFA", "#4ADE80", "#FB923C", "#38BDF8",
]

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"], family="Inter, Segoe UI, sans-serif", size=13),
        colorway=COLOR_SEQ,
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"], linecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"], linecolor=COLORS["grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor=COLORS["surface"], font_size=13),
    )
)

pio.templates["euroscout"] = PLOTLY_TEMPLATE
