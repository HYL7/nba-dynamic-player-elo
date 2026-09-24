"""Build career-level regular vs playoff Elo change-rate analysis.

This module is designed for broad all-player comparison rather than
superstar-only storytelling. It keeps each player's full career sample,
separates regular-season and playoff games, and computes per-game relative
Elo change rates inside each stage.

Outputs:
    results/analysis/4.2_regular_vs_playoff_shape/career_stage_change.csv
    results/analysis/4.2_regular_vs_playoff_shape/career_stage_change_summary.txt
    results/analysis/4.2_regular_vs_playoff_shape/logic.md

Usage:
    python scripts/build_career_stage_change_analysis.py
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
DB_PATH = RESULTS / "nba_elo.db"
OUT_DIR = RESULTS / "analysis" / "4.2_regular_vs_playoff_shape"

MIN_REGULAR_GAMES = 100
MIN_PLAYOFF_GAMES = 5


def build_logic_doc(path: Path) -> None:
    text = """# 4.2 Regular vs Playoff Shape

## Goal

Measure how a player's Elo tends to move in regular-season games versus
playoff games, using the full career sample instead of season-by-season
cuts that would discard too many players.

## Worldline

Main table uses `canonical_pk2`, so playoff games actively change Elo and
their impact is intentionally strong.

## Sample gates

- Regular-season games >= 100
- Playoff games >= 5

These are wide enough to keep broad coverage, but still remove the most
misleading tiny playoff samples.

## Core metrics

For each player:

- `regular_total_delta`: sum of `delta_adj` across all regular-season games
- `playoff_total_delta`: sum of `delta_adj` across all playoff games
- `regular_baseline_elo`: median pre-game Elo in regular-season games
- `playoff_baseline_elo`: median pre-game Elo in playoff games

Then define stage-level relative per-game change rates:

- `regular_change_rate = regular_total_delta / regular_games / regular_baseline_elo`
- `playoff_change_rate = playoff_total_delta / playoff_games / playoff_baseline_elo`

Interpretation:

- Positive: player's Elo tends to rise in that stage
- Negative: player's Elo tends to fall in that stage
- Larger magnitude: stage changes the player's Elo more aggressively

## Comparison signal

- `playoff_minus_regular = playoff_change_rate - regular_change_rate`

Interpretation:

- Positive: player tends to gain more Elo per playoff game than per regular-season game
- Negative: player tends to lose more, or gain less, in playoffs than in regular season

## Recommended 2D plot

- X axis: `regular_change_rate`
- Y axis: `playoff_change_rate`

Quadrants:

- Q1: rises in both stages
- Q2: falls in regular season, rises in playoffs
- Q3: falls in both
- Q4: rises in regular season, falls in playoffs

This is the main all-player mechanism view.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-regular", type=int, default=MIN_REGULAR_GAMES)
    parser.add_argument("--min-playoffs", type=int, default=MIN_PLAYOFF_GAMES)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"missing database: {DB_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_logic_doc(OUT_DIR / "logic.md")

    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            """
            SELECT player_id,
                   game_track,
                   rating_before,
                   delta_adj
            FROM variant_timeline
            WHERE variant_id = 'canonical_pk2'
            """,
            con,
        )
        players = pd.read_sql_query(
            "SELECT player_id, full_name FROM players",
            con,
        )
    finally:
        con.close()

    g = (
        df.groupby(["player_id", "game_track"])
        .agg(
            games=("delta_adj", "size"),
            total_delta=("delta_adj", "sum"),
            baseline_elo=("rating_before", "median"),
            mean_delta=("delta_adj", "mean"),
            median_delta=("delta_adj", "median"),
        )
        .reset_index()
    )

    wide = g.pivot(index="player_id", columns="game_track")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    out = wide.merge(players, on="player_id", how="left")

    out = out.rename(
        columns={
            "games_regular": "regular_games",
            "games_playoffs": "playoff_games",
            "total_delta_regular": "regular_total_delta",
            "total_delta_playoffs": "playoff_total_delta",
            "baseline_elo_regular": "regular_baseline_elo",
            "baseline_elo_playoffs": "playoff_baseline_elo",
            "mean_delta_regular": "regular_mean_delta",
            "mean_delta_playoffs": "playoff_mean_delta",
            "median_delta_regular": "regular_median_delta",
            "median_delta_playoffs": "playoff_median_delta",
        }
    )

    out = out[
        (out["regular_games"] >= args.min_regular)
        & (out["playoff_games"] >= args.min_playoffs)
    ].copy()

    out["regular_change_rate"] = (
        out["regular_total_delta"] / out["regular_games"] / out["regular_baseline_elo"]
    )
    out["playoff_change_rate"] = (
        out["playoff_total_delta"] / out["playoff_games"] / out["playoff_baseline_elo"]
    )
    out["playoff_minus_regular"] = out["playoff_change_rate"] - out["regular_change_rate"]
    out["quadrant"] = out.apply(
        lambda r: (
            "Q1" if r["regular_change_rate"] >= 0 and r["playoff_change_rate"] >= 0
            else "Q2" if r["regular_change_rate"] < 0 and r["playoff_change_rate"] >= 0
            else "Q3" if r["regular_change_rate"] < 0 and r["playoff_change_rate"] < 0
            else "Q4"
        ),
        axis=1,
    )

    ordered = out[
        [
            "player_id",
            "full_name",
            "regular_games",
            "playoff_games",
            "regular_baseline_elo",
            "playoff_baseline_elo",
            "regular_total_delta",
            "playoff_total_delta",
            "regular_mean_delta",
            "playoff_mean_delta",
            "regular_median_delta",
            "playoff_median_delta",
            "regular_change_rate",
            "playoff_change_rate",
            "playoff_minus_regular",
            "quadrant",
        ]
    ].sort_values(["playoff_minus_regular", "playoff_change_rate"], ascending=[False, False])

    suffix = f"_{args.tag}" if args.tag else ""
    out_csv = OUT_DIR / f"career_stage_change{suffix}.csv"
    ordered.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary = [
        f"players_kept={len(ordered)}",
        f"min_regular_games={MIN_REGULAR_GAMES}",
        f"min_playoff_games={MIN_PLAYOFF_GAMES}",
        "",
        "quadrant_counts:",
        ordered["quadrant"].value_counts().sort_index().to_string(),
        "",
        "top_playoff_minus_regular:",
        ordered.nlargest(15, "playoff_minus_regular")[
            ["full_name", "regular_change_rate", "playoff_change_rate", "playoff_minus_regular", "quadrant"]
        ].to_string(index=False),
        "",
        "bottom_playoff_minus_regular:",
        ordered.nsmallest(15, "playoff_minus_regular")[
            ["full_name", "regular_change_rate", "playoff_change_rate", "playoff_minus_regular", "quadrant"]
        ].to_string(index=False),
    ]
    summary_path = OUT_DIR / f"career_stage_change_summary{suffix}.txt"
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {summary_path}")
    print(f"wrote {OUT_DIR / 'logic.md'}")
    print(f"players_kept={len(ordered)}")


if __name__ == "__main__":
    main()
