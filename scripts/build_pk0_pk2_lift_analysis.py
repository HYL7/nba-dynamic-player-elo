"""Build canonical pk0 vs pk2 player-lift analysis table.

This script treats playoff_k=0 as the pure regular-season baseline and
playoff_k=2 as the strong-playoff worldline. It outputs one row per player
after applying minimum sample gates, so the result can be used directly for
2D scatter analysis.

Outputs:
    results/analysis/pk0_pk2_player_lift.csv
    results/analysis/pk0_pk2_player_lift_summary.txt

Usage:
    python scripts/build_pk0_pk2_lift_analysis.py
    python scripts/build_pk0_pk2_lift_analysis.py --min-regular 100 --min-playoffs 8
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = PROJECT_ROOT / "results"
ANALYSIS = RESULTS / "analysis"
DB_PATH = RESULTS / "nba_elo.db"

TOP_SHARE = 0.05
REL_EPS = 0.01


def ensure_db(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"missing database: {path}")


def load_variant_timeline(con: sqlite3.Connection, playoff_k: float) -> pd.DataFrame:
    query = """
        SELECT vt.variant_id,
               vt.playoff_k,
               vt.game_id,
               vt.game_date,
               vt.game_track,
               vt.player_id,
               vt.rating_after,
               p.full_name
        FROM variant_timeline vt
        JOIN players p
          ON p.player_id = vt.player_id
        WHERE vt.variant_id = ?
        ORDER BY vt.player_id, vt.game_date, vt.game_id
    """
    return pd.read_sql_query(query, con, params=(f"canonical_pk{playoff_k:g}",))


def load_game_counts(con: sqlite3.Connection) -> pd.DataFrame:
    query = """
        SELECT player_id,
               SUM(CASE WHEN game_track = 'regular' THEN 1 ELSE 0 END) AS regular_games,
               SUM(CASE WHEN game_track = 'playoffs' THEN 1 ELSE 0 END) AS playoff_games
        FROM variant_timeline
        WHERE variant_id = 'canonical_pk2'
        GROUP BY player_id
    """
    return pd.read_sql_query(query, con)


def summarize_variant(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    parts = []
    for player_id, g in df.groupby("player_id", sort=False):
        g = g.sort_values(["game_date", "game_id"])
        peak_idx = g["rating_after"].idxmax()
        peak_row = g.loc[peak_idx]
        top_n = max(1, int(math.ceil(len(g) * TOP_SHARE)))
        top_mean = g.nlargest(top_n, "rating_after")["rating_after"].mean()
        parts.append(
            {
                "player_id": int(player_id),
                "player_name": peak_row["full_name"],
                f"peak_elo_{prefix}": float(peak_row["rating_after"]),
                f"peak_date_{prefix}": str(peak_row["game_date"])[:10],
                f"peak_track_{prefix}": str(peak_row["game_track"]),
                f"high_end_mean_{prefix}": float(top_mean),
                f"n_games_{prefix}": int(len(g)),
            }
        )
    out = pd.DataFrame(parts)
    out = out.sort_values([f"peak_elo_{prefix}", "player_id"], ascending=[False, True]).reset_index(drop=True)
    out[f"peak_rank_{prefix}"] = out.index + 1
    return out


def classify(row: pd.Series) -> str:
    peak = float(row["relative_peak_lift"])
    high = float(row["relative_high_end_lift"])
    if peak <= -REL_EPS or high <= -REL_EPS:
        return "playoff_penalty"
    if peak >= REL_EPS and high >= REL_EPS:
        return "playoff_redefined"
    if peak >= REL_EPS and abs(high) < REL_EPS:
        return "playoff_spike"
    if abs(peak) < REL_EPS and high >= REL_EPS:
        return "steady_boost"
    return "playoff_neutral"


def source_switch_signal(row: pd.Series) -> int:
    a = row["peak_track_pk0"]
    b = row["peak_track_pk2"]
    if a == "regular" and b == "playoffs":
        return 1
    if a == "playoffs" and b == "regular":
        return -1
    return 0


def source_switch_label(row: pd.Series) -> str:
    return f"{row['peak_track_pk0']}_to_{row['peak_track_pk2']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-regular", type=int, default=200)
    parser.add_argument("--min-playoffs", type=int, default=15)
    args = parser.parse_args()

    ensure_db(DB_PATH)
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(DB_PATH)
    try:
        counts = load_game_counts(con)
        pk0 = summarize_variant(load_variant_timeline(con, 0.0), "pk0")
        pk2 = summarize_variant(load_variant_timeline(con, 2.0), "pk2")
    finally:
        con.close()

    merged = (
        counts.merge(pk0, on="player_id", how="inner")
        .merge(pk2, on=["player_id", "player_name"], how="inner")
    )
    merged = merged[
        (merged["regular_games"] >= args.min_regular)
        & (merged["playoff_games"] >= args.min_playoffs)
    ].copy()

    merged["peak_delta"] = merged["peak_elo_pk2"] - merged["peak_elo_pk0"]
    merged["high_end_delta"] = merged["high_end_mean_pk2"] - merged["high_end_mean_pk0"]
    merged["relative_peak_lift"] = merged["peak_delta"] / merged["peak_elo_pk0"]
    merged["relative_high_end_lift"] = merged["high_end_delta"] / merged["high_end_mean_pk0"]
    merged["peak_rank_change"] = merged["peak_rank_pk0"] - merged["peak_rank_pk2"]
    merged["peak_source_switch"] = merged.apply(source_switch_label, axis=1)
    merged["peak_source_switch_signal"] = merged.apply(source_switch_signal, axis=1)
    merged["analysis_class"] = merged.apply(classify, axis=1)
    merged["quadrant"] = merged.apply(
        lambda r: (
            "Q1" if r["relative_peak_lift"] >= 0 and r["relative_high_end_lift"] >= 0
            else "Q2" if r["relative_peak_lift"] < 0 and r["relative_high_end_lift"] >= 0
            else "Q3" if r["relative_peak_lift"] < 0 and r["relative_high_end_lift"] < 0
            else "Q4"
        ),
        axis=1,
    )

    ordered = merged[
        [
            "player_id",
            "player_name",
            "regular_games",
            "playoff_games",
            "peak_elo_pk0",
            "peak_elo_pk2",
            "peak_delta",
            "relative_peak_lift",
            "peak_rank_pk0",
            "peak_rank_pk2",
            "peak_rank_change",
            "peak_date_pk0",
            "peak_date_pk2",
            "peak_track_pk0",
            "peak_track_pk2",
            "peak_source_switch",
            "peak_source_switch_signal",
            "high_end_mean_pk0",
            "high_end_mean_pk2",
            "high_end_delta",
            "relative_high_end_lift",
            "analysis_class",
            "quadrant",
        ]
    ].sort_values(
        ["analysis_class", "relative_peak_lift", "relative_high_end_lift", "player_name"],
        ascending=[True, False, False, True],
    )

    out_csv = ANALYSIS / "pk0_pk2_player_lift.csv"
    ordered.to_csv(out_csv, index=False, encoding="utf-8-sig")

    class_counts = ordered["analysis_class"].value_counts().sort_index()
    summary_lines = [
        f"players_kept={len(ordered)}",
        f"min_regular={args.min_regular}",
        f"min_playoffs={args.min_playoffs}",
        "",
        "class_counts:",
    ]
    summary_lines.extend(f"  {k}: {v}" for k, v in class_counts.items())
    summary_lines.extend(
        [
            "",
            "top_relative_peak_lift:",
            ordered.nlargest(10, "relative_peak_lift")[
                ["player_name", "relative_peak_lift", "relative_high_end_lift", "analysis_class"]
            ].to_string(index=False),
            "",
            "bottom_relative_peak_lift:",
            ordered.nsmallest(10, "relative_peak_lift")[
                ["player_name", "relative_peak_lift", "relative_high_end_lift", "analysis_class"]
            ].to_string(index=False),
        ]
    )
    out_txt = ANALYSIS / "pk0_pk2_player_lift_summary.txt"
    out_txt.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {out_txt}")
    print(f"players_kept={len(ordered)}")
    print(class_counts.to_string())


if __name__ == "__main__":
    main()
