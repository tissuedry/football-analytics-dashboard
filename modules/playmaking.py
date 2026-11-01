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
        raise ValueError("Kolom wajib tidak ditemukan: " + ", ".join(missing))

    data = df.copy(deep=True).reset_index(drop=True)
    data["record_id"] = np.arange(len(data)) + 1
    notes = []

    for col in IDENTITY[:-1]:
        data[col] = data[col].astype("string").str.strip().replace("", pd.NA)
        data[col] = data[col].fillna("Unknown")

    data["pos"] = data["pos"].str.split(",").str[0].str.strip().str.upper()

    # Hilangkan baris total multi-tim agar tidak terjadi double counting.
    total_rows = data["team"].str.fullmatch(
        r"(?i)(?:\d+\s+(?:squads?|clubs?|teams?)|(?:all\s+)?(?:squads|clubs|teams)|total)",
        na=False,
    )
    if total_rows.any():
        notes.append(f"{int(total_rows.sum())} baris total multi-tim dikeluarkan untuk menghindari double counting.")
        data = data.loc[~total_rows].copy()

    numeric_cols = [
        "minutes", "key_passes", "prog_passes", "xag", "assists",
        "sca", "gca", "prog_carries", "cmp", "att", "cmp_pct",
    ]
    for col in numeric_cols:
        if col not in data.columns:
            data[col] = np.nan
            if col in METRICS:
                notes.append(f"Kolom {col} tidak tersedia; metrik terkait tidak dapat dianalisis.")
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
            notes.append(f"{col}: {int(invalid.sum())} nilai tidak valid diperlakukan sebagai data kosong.")
        data[col] = values.where(valid)

    # Hitung completion percentage hanya jika belum tersedia.
    calculated_pct = data["cmp"] / data["att"].where(data["att"] > 0) * 100
    data["pass_completion_pct"] = data["cmp_pct"].where(
        data["cmp_pct"].notna(), calculated_pct
    )

    # Rate per 90 menit. Nilai kosong tidak diubah menjadi nol.
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
    column = metric + "_per90" if mode == "Per 90 Menit" else metric
    ranked = data.dropna(subset=[column]).sort_values(
        [column, "player", "team", "league", "record_id"],
        ascending=[False, True, True, True, True],
        kind="stable",
    ).head(top_n)

    if ranked.empty:
        st.info("Tidak ada data yang tersedia untuk metrik dan filter ini.")
        return

    unit = "per 90 menit" if mode == "Per 90 Menit" else "total"
    fig = px.bar(
        ranked.sort_values(column),
        x=column,
        y="label",
        orientation="h",
        text=column,
        color="pos",
        color_discrete_map={k: COLORS.get(k, COLORS["muted"]) for k in ["FW", "MF", "DF", "GK"]},
        hover_data={"team": True, "league": True, "minutes": ":.0f", "pos": True},
        labels={column: f"{METRICS[metric]} ({unit})", "label": "Pemain", "pos": "Posisi"},
        title=f"Top {len(ranked)} Pemain — {METRICS[metric]}",
    )
    fig.update_traces(texttemplate="%{x:.2f}" if mode == "Per 90 Menit" else "%{x:.0f}", textposition="outside")
    _show_chart(fig, f"pm_rank_{metric}_{mode}")


def render(df: pd.DataFrame):
    """Render the complete Passing & Playmaking tab."""
    st.subheader("Passing & Playmaking")
    st.caption(
        "Analisis kreativitas, distribusi bola, dan kontribusi pemain dalam membangun peluang. "
        "Seluruh analisis mengikuti filter liga, posisi, dan menit bermain pada sidebar."
    )

    if df is None or df.empty:
        st.info("Tidak ada pemain yang sesuai dengan filter global. Silakan ubah filter di sidebar.")
        return

    try:
        data, notes = _prepare_data(df)
    except ValueError as exc:
        st.error(str(exc))
        return

    if data.empty:
        st.info("Tidak ada data pemain yang dapat dianalisis.")
        return

    valid_minutes = data["minutes"].dropna()
    if notes:
        with st.expander("Validasi dan ketersediaan data"):
            for note in notes:
                st.write("• " + note)

    # ---------------- KPI ----------------
    st.markdown("#### Ringkasan Playmaking")
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
        f"{len(data):,} player–team records dianalisis · "
        f"{data['player'].nunique():,} nama pemain · "
        f"{valid_minutes.sum():,.0f} total menit valid."
    )

    # ---------------- Ranking ----------------
    st.divider()
    st.markdown("#### 1. Playmaking Leaderboard")
    st.caption(
        "Pilih metrik untuk melihat pemain dengan kontribusi playmaking tertinggi. "
        "Per 90 menit membantu membandingkan pemain dengan waktu bermain berbeda."
    )
    controls = st.columns(3)
    selected_metric = controls[0].selectbox(
        "Metrik ranking",
        list(METRICS.keys()),
        format_func=METRICS.get,
        key="pm_metric",
    )
    ranking_mode = controls[1].radio(
        "Satuan ranking", ["Per 90 Menit", "Total"], horizontal=True, key="pm_mode"
    )
    top_n = controls[2].selectbox("Jumlah pemain", [5, 10, 15, 20], index=1, key="pm_top_n")
    _ranking_chart(data, selected_metric, top_n, ranking_mode)

    # ---------------- Creative profile ----------------
    st.divider()
    st.markdown("#### 2. Kreator Peluang Terbaik")
    st.caption(
        "Grafik membandingkan Key Passes, xAG, dan Assists. "
        "Gunakan mode per 90 menit untuk mengurangi pengaruh perbedaan waktu bermain."
    )
    creative_mode = st.radio(
        "Satuan grafik kreativitas",
        ["Per 90 Menit", "Total"],
        horizontal=True,
        key="pm_creative_mode",
    )
    creative_metrics = ["key_passes", "xag", "assists"]
    value_columns = [
        m + "_per90" if creative_mode == "Per 90 Menit" else m
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
        var_name="Metrik",
        value_name="Nilai",
    ).dropna(subset=["Nilai"])

    if creative_long.empty:
        st.info("Data kreativitas belum tersedia.")
    else:
        # Ambil top pemain berdasarkan gabungan ranking metrik agar grafik tetap terbaca.
        top_players = (
            creative.groupby("label")[["Key Passes", "xAG", "Assists"]]
            .sum(min_count=1).fillna(0).sum(axis=1)
            .nlargest(top_n).index
        )
        creative_long = creative_long[creative_long["label"].isin(top_players)]
        fig = px.bar(
            creative_long,
            x="player",
            y="Nilai",
            color="Metrik",
            barmode="group",
            hover_data=["team", "pos", "minutes"],
            color_discrete_sequence=COLOR_SEQ,
            labels={"player": "Pemain", "Nilai": "Nilai", "Metrik": "Metrik"},
            title=f"Perbandingan Kreativitas Pemain ({creative_mode})",
        )
        _show_chart(fig, "pm_creative_chart", 480)

    # ---------------- Key passes vs xAG ----------------
    st.divider()
    st.markdown("#### 3. Hubungan Key Passes dengan Expected Assisted Goals")
    st.caption(
        "Setiap titik mewakili satu pemain. Pemain di kanan memiliki lebih banyak key passes; "
        "pemain di atas memiliki nilai xAG lebih tinggi. Korelasi menunjukkan hubungan, bukan sebab-akibat."
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
                "pos": "Posisi",
                "sca": "SCA",
            },
            title="Key Passes per 90 vs xAG per 90",
        )
        _show_chart(fig, "pm_keypasses_xag", 480)
        corr = scatter["key_passes_per90"].corr(scatter["xag_per90"])
        st.metric("Korelasi Pearson Key Passes dan xAG", f"{corr:.3f}" if pd.notna(corr) else "—")
    else:
        st.info("Minimal dua pemain dengan data Key Passes dan xAG diperlukan untuk membuat scatter plot.")

    # ---------------- Passing accuracy ----------------
    st.divider()
    st.markdown("#### 4. Passing Accuracy")
    st.caption(
        "Persentase operan sukses dihitung dari operan berhasil dibagi total percobaan operan "
        "jika persentase sumber tidak tersedia."
    )
    passing = data.dropna(subset=["pass_completion_pct"]).copy()
    passing = passing.sort_values(
        ["pass_completion_pct", "player"], ascending=[False, True]
    ).head(top_n)
    if passing.empty:
        st.info("Data passing completion tidak tersedia.")
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
            labels={"pass_completion_pct": "Pass Completion (%)", "label": "Pemain", "pos": "Posisi"},
            title=f"Top {len(passing)} Pemain berdasarkan Akurasi Operan",
        )
        fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
        _show_chart(fig, "pm_passing_accuracy")

    # ---------------- Player table ----------------
    st.divider()
    st.markdown("#### 5. Statistik Playmaking Pemain")
    search = st.text_input("Cari nama pemain atau tim", key="pm_search").strip().lower()
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
                f"- **{title}:** {best['player']} ({best['team']}) mencatat "
                f"{best[metric]:.2f}."
            )

    if insights:
        st.markdown("\n".join(insights))
    else:
        st.info("Belum cukup data untuk menghasilkan insight otomatis.")

    below_threshold = int(data["minutes"].lt(SMALL_SAMPLE_MINUTES).sum())
    st.caption(
        f"Catatan interpretasi: {below_threshold} dari {len(data)} record memiliki menit bermain "
        f"di bawah {SMALL_SAMPLE_MINUTES}. Statistik per 90 dari sampel menit yang kecil "
        "dapat berubah secara signifikan. Angka kreativitas mengukur volume/keterlibatan, "
        "bukan kualitas keputusan pemain secara keseluruhan."
    )
