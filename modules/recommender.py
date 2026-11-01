"""Member 4 — ML Scouting & Similarity.

"Find players similar to [Player X]" feature across leagues:
- Features are normalized per-90-minutes then scaled (StandardScaler).
- Similarity is computed with Cosine Similarity between players in the same
  position group (FW/MF/DF/GK are compared within their own group because
  their statistical profiles aren't comparable across positions).
- K-Means is used to group players within the same position into several
  "role types" and give extra context beyond the similarity list.
- The model & similarity matrix are wrapped in @st.cache_resource so they
  aren't recomputed on every click.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from utils.theme import COLORS, COLOR_SEQ, PLOTLY_TEMPLATE

# features per position group — all computed per-90-minutes before scaling
POSITION_FEATURES = {
    "FW": ["goals", "xg", "npxg", "shots", "shots_on_target", "sca", "gca", "prog_carries", "key_passes"],
    "MF": ["xag", "key_passes", "prog_passes", "prog_carries", "sca", "gca", "tackles", "interceptions"],
    "DF": ["tackles", "interceptions", "clearances", "recoveries", "prog_passes", "prog_carries", "tackles_interceptions"],
    "GK": ["saves", "save_pct", "cs_pct", "ga90", "psxg"],
}

MIN_NINETIES_FOR_SIMILARITY = 5  # ~450 minutes, a looser threshold than the global filter

# human-readable labels for each raw feature column name, used on the
# radar chart axes and cluster summary (avoids raw FBref abbreviations)
FEATURE_LABELS = {
    "goals": "Goals",
    "xg": "Expected Goals (xG)",
    "npxg": "Non-Penalty xG",
    "shots": "Shots",
    "shots_on_target": "Shots on Target",
    "sca": "Shot-Creating Actions",
    "gca": "Goal-Creating Actions",
    "prog_carries": "Progressive Carries",
    "prog_passes": "Progressive Passes",
    "key_passes": "Key Passes",
    "xag": "Expected Assisted Goals (xAG)",
    "tackles": "Tackles",
    "interceptions": "Interceptions",
    "tackles_interceptions": "Tackles + Interceptions",
    "clearances": "Clearances",
    "recoveries": "Ball Recoveries",
    "saves": "Saves",
    "save_pct": "Save %",
    "cs_pct": "Clean Sheet %",
    "ga90": "Goals Against / 90",
    "psxg": "Post-Shot xG (PSxG)",
}


def _per90(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    nineties = df["nineties"].replace(0, np.nan)
    out = pd.DataFrame(index=df.index)
    for col in cols:
        if col not in df.columns:
            out[col] = 0.0
            continue
        if col.endswith("_pct") or col == "ga90":
            out[col] = df[col]  # already a ratio/average, no need to divide by minutes
        else:
            # psxg in the raw CSV is cumulative for the season (correlation 0.93 with
            # minutes played), NOT a rate — must be divided by minutes like saves/shots,
            # otherwise it becomes a proxy for "how long they played" not "keeper quality"
            out[col] = (df[col] / nineties).fillna(0.0)
    return out.fillna(0.0)


def _optimal_k(scaled: np.ndarray, k_min: int = 2, k_max: int = 8):
    """Determine the number of clusters via two methods:
    - Elbow method: inertia (WCSS) per k, shown as a visual cross-check
      (a sharp drop = clusters still separate distinct groups; it flattening
      out = groups are starting to look alike).
    - Silhouette score: for each k, measures how tightly each player fits its
      own cluster versus the nearest neighboring cluster (range -1..1, higher
      is better). This is a more objective metric than visually looking for
      an 'elbow', so it's used as the FINAL DECIDER for k —
      Elbow is still shown for comparison, not as the decider.
    """
    k_max = min(k_max, scaled.shape[0] - 1)
    if k_max < k_min:
        return max(2, scaled.shape[0] - 1), {}, {}

    inertias = {}
    silhouettes = {}
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled)
        inertias[k] = labels.inertia_
        silhouettes[k] = silhouette_score(scaled, labels.labels_)

    best_k = max(silhouettes, key=silhouettes.get)
    return best_k, inertias, silhouettes


def _pca_for_clustering(scaled: np.ndarray, variance_threshold: float = 0.8):
    """PCA used only for the CLUSTERING stage (not for similarity/radar chart,
    which still uses the raw features so each stat stays individually
    readable). The raw features used here are often highly correlated with
    each other (e.g. Goals, xG, npxG, Shots, Shots on Target all represent
    the 'finishing productivity' concept) — without PCA, one concept gets
    counted multiple times in KMeans's Euclidean distance, artificially
    inflating its weight compared to concepts represented by fewer features
    (e.g. chance creation). PCA builds new, mutually independent axes (PC1,
    PC2, ...); we take just enough components to cover `variance_threshold`
    of the total variance, discarding the rest as noise.
    """
    n_features = scaled.shape[1]
    max_components = min(n_features, scaled.shape[0] - 1)
    pca_full = PCA(n_components=max_components, random_state=42).fit(scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = max(2, min(int(np.searchsorted(cumvar, variance_threshold) + 1), max_components))

    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(scaled)
    return reduced, pca.explained_variance_ratio_, n_components


@st.cache_resource
def _build_similarity(df: pd.DataFrame, pos: str):
    """Scaler + cosine similarity matrix for one position group."""
    features = POSITION_FEATURES[pos]
    subset = df[df["pos"] == pos].copy()
    subset = subset[subset["nineties"] >= MIN_NINETIES_FOR_SIMILARITY]
    if len(subset) < 2:
        return subset, None, None, features, 0, {}, {}, 0, np.array([])

    feat_df = _per90(subset, features)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feat_df.values)
    sim_matrix = cosine_similarity(scaled)  # similarity & radar stay in the original feature space

    pca_scaled, pca_var_ratio, n_components = _pca_for_clustering(scaled)
    n_clusters, inertias, silhouettes = _optimal_k(pca_scaled)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    subset = subset.reset_index(drop=True)
    subset["cluster"] = kmeans.fit_predict(pca_scaled)

    return subset, sim_matrix, scaled, features, n_clusters, inertias, silhouettes, n_components, pca_var_ratio


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _radar_chart(subset, feat_df, idx_a, idx_b, label_a, label_b):
    cats = [FEATURE_LABELS.get(c, c) for c in feat_df.columns.tolist()]
    row_a = feat_df.iloc[idx_a]
    row_b = feat_df.iloc[idx_b]

    # percentile rank (0-100) against ALL players in the same position group, not
    # "percentage of the max value" — avoids a single outlier compressing all
    # other players down to misleadingly small numbers.
    percentile_df = feat_df.rank(pct=True) * 100
    norm_a = percentile_df.iloc[idx_a].tolist()
    norm_b = percentile_df.iloc[idx_b].tolist()

    hover = (
        "<b>%{theta}</b><br>Value: %{customdata:.2f} per 90<br>"
        "Percentile rank: %{r:.0f} (higher than %{r:.0f}%% of same-position peers)"
        "<extra>%{fullData.name}</extra>"
    )
    raw_a = row_a.tolist()
    raw_b = row_b.tolist()

    marker_style = dict(size=9, line=dict(width=1.5, color=COLORS["background"]))

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm_a + [norm_a[0]], theta=cats + [cats[0]],
        customdata=raw_a + [raw_a[0]], hovertemplate=hover, hoveron="points",
        mode="lines+markers", marker=marker_style,
        fill="toself", fillcolor=_hex_to_rgba(COLOR_SEQ[0], 0.18),
        name=label_a, line_color=COLOR_SEQ[0],
    ))
    fig.add_trace(go.Scatterpolar(
        r=norm_b + [norm_b[0]], theta=cats + [cats[0]],
        customdata=raw_b + [raw_b[0]], hovertemplate=hover, hoveron="points",
        mode="lines+markers", marker=marker_style,
        fill="toself", fillcolor=_hex_to_rgba(COLOR_SEQ[3], 0.18),
        name=label_b, line_color=COLOR_SEQ[3],
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        hovermode="closest",
        polar=dict(
            bgcolor=COLORS["surface"],
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                showticklabels=True,
                angle=90,
                tickfont=dict(color=COLORS["muted"], size=11),
                gridcolor=COLORS["grid"],
                linecolor=COLORS["grid"],
            ),
            angularaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["grid"]),
            domain=dict(x=[0.08, 0.92], y=[0.05, 0.95]),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=650,
        margin=dict(l=40, r=40, t=80, b=20),
        title="Stat profile comparison (relative to players in the same position group, per 90 minutes)",
    )
    return fig


def _describe_clusters(subset, scaled, features, top_k=2):
    """For each cluster, find the features with the highest average z-score so
    the user knows which playing style stands out — not just a 0-4 number label."""
    scaled_df = pd.DataFrame(scaled, columns=features)
    scaled_df["cluster"] = subset["cluster"].values
    cluster_means = scaled_df.groupby("cluster")[features].mean()

    rows = []
    counts = subset["cluster"].value_counts()
    for cluster_id, row in cluster_means.iterrows():
        top_features = row.sort_values(ascending=False).head(top_k).index.tolist()
        traits = ", ".join(FEATURE_LABELS.get(f, f) for f in top_features)
        rows.append({
            "Cluster": int(cluster_id),
            "Players": int(counts.get(cluster_id, 0)),
            "Stands out in": traits,
        })
    return pd.DataFrame(rows).sort_values("Cluster").reset_index(drop=True)


def render(df):
    st.subheader("ML Scouting & Similarity")
    st.caption(
        "Find players with the closest statistical profile — computed per position, "
        "normalized per 90 minutes, across leagues."
    )

    available_pos = [p for p in ["FW", "MF", "DF", "GK"] if (df["pos"] == p).sum() >= 2]
    if not available_pos:
        st.warning("Not enough players under the current filters for similarity analysis.")
        return

    col_pos, col_player = st.columns([1, 2])
    with col_pos:
        pos_choice = st.selectbox("Position", available_pos)

    subset, sim_matrix, scaled, features, n_clusters, inertias, silhouettes, n_components, pca_var_ratio = _build_similarity(df, pos_choice)

    if sim_matrix is None:
        st.warning(f"Not enough {pos_choice} players (minutes ≥ {MIN_NINETIES_FOR_SIMILARITY * 90}) under the current filters.")
        return

    player_options = subset["player"] + " (" + subset["team"] + ", " + subset["league"] + ")"
    with col_player:
        selected_label = st.selectbox("Reference player", player_options.tolist())
    selected_idx = player_options.tolist().index(selected_label)

    top_n = st.slider("Number of recommendations", 3, 15, 5)

    sims = sim_matrix[selected_idx]
    order = np.argsort(-sims)
    order = [i for i in order if i != selected_idx][:top_n]

    result = subset.iloc[order][["player", "team", "league", "nineties"]].copy()
    result["similarity"] = [round(float(sims[i]), 3) for i in order]
    result = result.rename(columns={
        "player": "Player", "team": "Team", "league": "League",
        "nineties": "Full Matches Played (90-min equiv.)", "similarity": "Similarity Score",
    })
    st.dataframe(result, use_container_width=True, hide_index=True)

    if order:
        feat_df = _per90(subset, features)
        compare_labels = [player_options.tolist()[i] for i in order]
        compare_label = st.selectbox("Compare reference player against", compare_labels)
        compare_idx = order[compare_labels.index(compare_label)]
        fig = _radar_chart(
            subset, feat_df, selected_idx, compare_idx,
            subset.iloc[selected_idx]["player"], subset.iloc[compare_idx]["player"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Playing styles (K-Means clusters) for this position"):
        st.caption(
            f"K-Means grouped players at this position into **{n_clusters} clusters** purely "
            "from their statistical profile (no inherent names — cluster IDs are arbitrary). "
            "The number of clusters was chosen automatically — see the methods below."
        )
        if n_components:
            explained = pca_var_ratio.sum() * 100
            st.caption(
                f"Clustering runs on **{n_components} PCA components** (not the raw "
                f"{len(features)} stats directly), capturing {explained:.0f}% of the "
                "variance. Several raw stats are highly correlated (e.g. Goals, xG, "
                "Shots all reflect 'finishing output') — without PCA those correlated "
                "stats get double-counted in the distance KMeans uses, inflating their "
                "weight. PCA combines them into independent axes first, so clustering "
                "reflects distinct underlying concepts rather than how many raw columns "
                "happen to measure the same thing."
            )
        cluster_summary = _describe_clusters(subset, scaled, features)
        st.dataframe(cluster_summary, use_container_width=True, hide_index=True)
        st.bar_chart(subset["cluster"].value_counts().sort_index())

        if silhouettes:
            best_score = silhouettes[n_clusters]
            st.markdown("**Silhouette score — the deciding metric**")
            st.caption(
                "For every k, measures how well players fit their own cluster vs. the "
                "nearest other cluster (range -1 to 1; higher = tighter, better-separated "
                f"groups). k = **{n_clusters}** was picked because it has the highest "
                f"average silhouette score ({best_score:.3f}) — this is the metric that "
                "actually decides the cluster count, not just a visual guess."
            )
            sil_df = pd.DataFrame({
                "k (number of clusters)": list(silhouettes.keys()),
                "Silhouette score": list(silhouettes.values()),
            }).set_index("k (number of clusters)")
            st.line_chart(sil_df)

        if inertias:
            st.markdown("**Elbow method — visual cross-check**")
            st.caption(
                "A steep drop means adding another cluster still separates clearly "
                "different groups of players; a flat section means the groups are "
                "already similar. Shown here for comparison — the silhouette score "
                "above is what actually determines k."
            )
            elbow_df = pd.DataFrame({
                "k (number of clusters)": list(inertias.keys()),
                "Inertia (WCSS)": list(inertias.values()),
            }).set_index("k (number of clusters)")
            st.line_chart(elbow_df)

    top_match = subset.iloc[order[0]]["player"] if order else None
    st.markdown("**Key Insight**")
    if top_match:
        st.info(
            f"The player with the closest statistical profile to **{selected_label}** at {pos_choice} "
            f"is **{top_match}** (similarity score {result.iloc[0]['Similarity Score']}). "
            "Use this to find a cross-league replacement with a comparable playing style."
        )
