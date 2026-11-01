"""Player data loader & cleaner for EuroScout AI.

Source: data/players_data-2024_2025_cleansing.csv (FBref, Europe's Big 5
leagues 2024-2025). The raw columns are a merge of several FBref stat tables
(standard, shooting, passing, passing_types, gca, defense, possession,
playing_time, misc, keeper, keeper_adv), so many column names repeat and are
already disambiguated via a suffix (e.g. `Tkl` vs `Tkl_stats_defense`).

load_data() does the following:
1. Drops raw index columns (`source_row`, `Rk`).
2. Normalizes ALL column names to lowercase snake_case
   (e.g. `xAG` -> `xag`, `Tkl+Int` -> `tkl_int`, `SoT%` -> `sot_pct`).
3. Renames core columns to standard terms shared across modules via RENAME_MAP
   (player, team, league, pos, minutes, goals, assists, xg, xag, etc.).
4. Adds `pos`: simplifies the raw `Pos` value (can be multi-position, e.g.
   "MF,FW") down to a single primary label FW / MF / DF / GK.
5. Adds `league`: cleans the raw `Comp` value (e.g. "eng Premier League"
   -> "Premier League").

Other modules MUST NOT read the CSV themselves — always call load_data().
"""

import os
import re

import pandas as pd
import streamlit as st

DATA_PATH = "data/players_data.csv"

# raw column (after snake_case) -> standard name used across all modules
RENAME_MAP = {
    "player": "player",
    "nation": "nation",
    "squad": "team",
    "comp": "league_raw",
    "pos": "pos_raw",
    "age": "age",
    "min": "minutes",
    "90s": "nineties",
    "gls": "goals",
    "ast": "assists",
    "g_plus_a": "goals_assists",
    "xg": "xg",
    "npxg": "npxg",
    "xag": "xag",
    "npxg_plus_xag": "npxg_xag",
    "sh": "shots",
    "sot": "shots_on_target",
    "prgc": "prog_carries",
    "prgp": "prog_passes",
    "prgr": "prog_receptions",
    "kp": "key_passes",
    "sca": "sca",
    "sca90": "sca_per90",
    "gca": "gca",
    "gca90": "gca_per90",
    "tkl": "tackles",
    "int": "interceptions",
    "tkl_plus_int": "tackles_interceptions",
    "clr": "clearances",
    "err": "errors",
    "recov": "recoveries",
    "touches": "touches",
}

_POS_MAP = {"FW": "FW", "MF": "MF", "DF": "DF", "GK": "GK"}


def _snake_case(col: str) -> str:
    col = col.strip()
    col = col.replace("%", "_pct").replace("+/-", "_net").replace("+", "_plus_")
    col = col.replace("/", "_per_")
    col = re.sub(r"[^0-9a-zA-Z]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col.lower()


def _dedupe_columns(cols: list) -> list:
    """Some raw columns can still collide after snake_case (e.g. 'xG' vs
    'xG+/-' -> 'xg' vs 'xg_net'). The first column is kept as-is;
    later duplicates get a _2, _3, ... suffix so the DataFrame doesn't error."""
    seen = {}
    result = []
    for c in cols:
        if c not in seen:
            seen[c] = 1
            result.append(c)
        else:
            seen[c] += 1
            result.append(f"{c}_{seen[c]}")
    return result


def _simplify_pos(raw) -> str:
    """Get the primary position from the raw Pos value (e.g. 'MF,FW' -> 'MF')."""
    if not isinstance(raw, str) or not raw:
        return "DF"
    first = raw.split(",")[0].strip()
    return _POS_MAP.get(first, "DF")


def _simplify_league(raw):
    """Strip the leading country code from the league name (e.g. 'eng Premier League' -> 'Premier League')."""
    if not isinstance(raw, str):
        return raw
    parts = raw.split(" ", 1)
    return parts[1] if len(parts) == 2 else raw


@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.drop(columns=[c for c in ("source_row", "Rk") if c in df.columns])

    df.columns = _dedupe_columns([_snake_case(c) for c in df.columns])
    df = df.rename(columns=RENAME_MAP)

    df["pos"] = df["pos_raw"].apply(_simplify_pos)
    df["league"] = df["league_raw"].apply(_simplify_league)
    df = df.copy()  # defragment after several column inserts above

    numeric_cols = [
        "age", "minutes", "nineties", "goals", "assists", "goals_assists", "xg", "npxg", "xag", "npxg_xag", "shots", "shots_on_target", "sca", "sca_per90", "gca", "gca_per90", "key_passes", "cmp", "att", "cmp_pct",
        "prog_carries", "prog_passes", "prog_receptions", "tackles",
        "interceptions", "tackles_interceptions", "clearances", "blocks", "recoveries", "def_3rd", "mid_3rd", "att_3rd", "err",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
