"""Full-window Elo run with the recommended D-023 parameters.

Loads every usable regular boxscore row from clean_data, runs the engine
with K=45 / surprise_scale=285 / surprise_anchor=game / draft init, writes
per-season diagnostics plus the full updates frame as Parquet, and checks
the D-021 invariants (row counts, per-game zero sum, no NaN).

Usage:
    python scripts/run_full_elo.py [start_season] [end_season] [playoff_k_multiplier] [alpha_modern] [scale] [on_court_mode]

Defaults are the locked 1.0 canonical track (K=45, theta=0, scale=285,
boost=6, tau=60, H=70, on_court_mode=raw). For the 0.9 track, pass
alpha_modern=0.9 scale=318 on_court_mode=raw.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.elo.engine import EloEngine

RESULTS = Path("results")

BOX_COLS = [
    "game_id", "game_date", "season", "season_id", "track",
    "personId", "playerteamId", "home", "win", "minutes",
    "scaled_minutes", "minutes_share", "plus_minus_available",
    "usable_game", "plusMinusPoints", "points", "assists", "blocks",
    "steals", "fieldGoalsAttempted", "fieldGoalsMade",
    "fieldGoalsPercentage", "threePointersAttempted", "threePointersMade",
    "threePointersPercentage", "freeThrowsAttempted", "freeThrowsMade",
    "freeThrowsPercentage", "reboundsDefensive", "reboundsOffensive",
    "reboundsTotal", "foulsPersonal", "turnovers",
]


def load_usable_boxscores(start_season, end_season):
    chunks = []
    for chunk in pd.read_csv(
        "clean_data/boxscores_clean.csv",
        usecols=BOX_COLS,
        chunksize=200000,
        low_memory=False,
    ):
        chunk = chunk[chunk["usable_game"] == 1]
        chunk = chunk[(chunk["season"] >= start_season) & (chunk["season"] <= end_season)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def build_draft_inputs(rookie_start=1425.0, pick1_rating=None, pick_slope=None, rookie_floor=None):
    # D-032: draft prior is a single uniform rookie_start; the legacy
    # pick1/slope/floor arguments are kept only as explicit overrides.
    if pick1_rating is None:
        pick1_rating = float(rookie_start)
    if pick_slope is None:
        pick_slope = 0.0
    if rookie_floor is None:
        rookie_floor = float(rookie_start)
    players = pd.read_csv("clean_data/players_clean.csv", low_memory=False)
    draft = pd.read_csv(
        "processed_data/draft_history.csv",
        usecols=["person_id", "overall_pick"],
        low_memory=False,
    )
    draft = draft.dropna(subset=["overall_pick"])
    picks = draft.groupby("person_id")["overall_pick"].min()
    players = players.merge(
        picks.rename("overall_pick"), left_on="personId", right_index=True, how="left"
    )

    rating_map = {}
    for row in players.itertuples(index=False):
        pid = int(row.personId)
        if pd.isna(row.overall_pick):
            rating_map[pid] = float(rookie_floor)
            continue
        rating = float(pick1_rating) - (float(row.overall_pick) - 1.0) * float(pick_slope)
        rating_map[pid] = max(rating, float(rookie_floor))
    return rating_map


def season_metrics(updates, box):
    minutes_map = box[["game_id", "personId", "scaled_minutes"]].rename(
        columns={"personId": "player_id"}
    )
    updates = updates.merge(minutes_map, on=["game_id", "player_id"], how="left")

    rows = []
    for season, grp in updates.groupby("season"):
        final = grp.sort_values(["game_date", "game_id"]).groupby("player_id").tail(1)
        minutes = (
            grp.groupby("player_id")["scaled_minutes"].sum()
            .rename("season_minutes")
            .reset_index()
        )
        merged = final.merge(minutes, on="player_id")
        ge1000 = merged[merged["season_minutes"] >= 1000]
        rows.append({
            "season": season,
            "n_players": len(final),
            "mean": final["rating_after"].mean(),
            "minutes_wmean": np.average(
                final["rating_after"], weights=merged["season_minutes"]
            ),
            "sd": final["rating_after"].std(ddof=0),
            "mean_ge1000min": ge1000["rating_after"].mean() if len(ge1000) else np.nan,
            "top_rating": final["rating_after"].max(),
        })
    return pd.DataFrame(rows)


def main():
    start_season = int(sys.argv[1]) if len(sys.argv) > 1 else 1976
    end_season = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    playoff_k = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    alpha_modern = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    scale = float(sys.argv[5]) if len(sys.argv) > 5 else 285.0
    on_court_mode = sys.argv[6] if len(sys.argv) > 6 else "raw"
    if on_court_mode not in {"raw", "team_relative", "on_off"}:
        raise ValueError(f"unknown on_court_mode: {on_court_mode}")
    suffix = f"{start_season}-{end_season}"

    t0 = time.time()
    print(f"loading usable boxscores {start_season}-{end_season}")
    box = load_usable_boxscores(start_season, end_season)
    print(f"loaded {len(box)} rows, {box['game_id'].nunique()} games "
          f"({time.time() - t0:.1f}s)")

    draft_ratings = build_draft_inputs(rookie_start=1425.0)
    print(f"draft ratings: {len(draft_ratings)}")

    engine = EloEngine(
        k=45.0, theta=0.0, home_advantage=70.0, alpha=1.0,
        rookie_boost=6.0, rookie_tau=60.0,
        init_mode="draft", draft_ratings=draft_ratings,
        rookie_ids=None, record_states=False,
        surprise_scale=scale, surprise_anchor="game",
        playoff_k_multiplier=playoff_k,
        alpha_modern=alpha_modern, alpha_modern_start_season=1996,
        on_court_mode=on_court_mode,
    )
    t1 = time.time()
    updates, _, inits = engine.run(box)
    elapsed = time.time() - t1
    print(f"engine ran in {elapsed:.1f}s: {len(updates)} updates, {len(inits)} inits")

    per_game = updates.groupby("game_id")["delta_adj"].sum()
    zero_sum_max = float(per_game.abs().max())
    nan_count = int(updates[
        ["rating_before", "rating_after", "perf_i", "perf_used", "delta_adj"]
    ].isna().sum().sum())
    games_run = int(updates["game_id"].nunique())
    print(f"max |sum(delta_adj)| per game: {zero_sum_max:.3e}")
    print(f"NaN in key columns: {nan_count}")
    print(f"updates per game: {len(updates) / games_run:.2f}")

    metrics = season_metrics(updates, box)
    RESULTS.mkdir(exist_ok=True)
    metrics.to_csv(RESULTS / f"full_elo_diagnostics_{suffix}.csv", index=False)

    # Transition check around the on-court era boundary (1996-97 is the
    # first real on-court season; alpha switches to alpha_modern there).
    boundary = metrics[metrics["season"].between(1994, 1997)]
    print("\n1994-1997 transition:")
    print(boundary.to_string(index=False))

    updates_out = RESULTS / f"full_elo_updates_{suffix}.parquet"
    updates.to_parquet(updates_out, index=False)
    print(f"wrote {updates_out} ({updates_out.stat().st_size / 1e6:.0f} MB)")

    summary = {
        "window": f"{start_season}-{end_season}",
        "games_run": games_run,
        "updates": len(updates),
        "engine_seconds": round(elapsed, 1),
        "max_abs_zero_sum": zero_sum_max,
        "nan_key_columns": nan_count,
        "params": "k=45,theta=0,H=70,alpha=1,boost=6,tau=60,"
                  f"init=draft,surprise_scale={scale:g},surprise_anchor=game,"
                  f"on_court_mode={on_court_mode},"
                  f"playoff_k_multiplier={playoff_k:g},"
                  f"alpha_modern={alpha_modern:g}",
    }
    summary_out = RESULTS / f"full_elo_summary_{suffix}.csv"
    pd.DataFrame([summary]).to_csv(summary_out, index=False)
    print(f"wrote {summary_out}")
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
