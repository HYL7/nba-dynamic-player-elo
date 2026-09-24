"""Calibrate rookie_start against the minutes-weighted exit ratings.

Runs the final parameter set over 1976-2025 with a uniform rookie start,
then for every season computes the final rating distribution of players who
do not appear in the next season.  The long-run minutes-weighted exit rating
is a relative entry-exit flow check (E-020); because updates are translation
invariant, the exit gap does not select the absolute level.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_full_elo import build_draft_inputs, load_usable_boxscores
from src.elo.engine import EloEngine

RESULTS = Path("results")
ROOKIE_START = float(sys.argv[1]) if len(sys.argv) > 1 else 1425.0


def final_ratings(updates):
    rows = (
        updates.sort_values(["game_date", "game_id"])
        .groupby(["season", "player_id"])
        .tail(1)[["season", "player_id", "rating_after"]]
        .copy()
    )
    minutes = (
        updates.groupby(["season", "player_id"])["minutes_share"]
        .sum()
        .rename("minutes")
    )
    return rows.merge(minutes, on=["season", "player_id"], how="left")


def exit_players(final):
    players_by_season = {
        int(s): set(grp["player_id"])
        for s, grp in final.groupby("season")
    }
    seasons = sorted(players_by_season)
    frames = []
    for s in seasons[:-1]:
        exits = final[final["season"] == s]
        exits = exits[~exits["player_id"].isin(players_by_season[s + 1])]
        if not exits.empty:
            frames.append(exits.assign(exit_season=s))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def exit_stats(final):
    exits = exit_players(final)
    rows = []
    for s, grp in exits.groupby("exit_season"):
        wmean = (
            float(np.average(grp["rating_after"], weights=grp["minutes"]))
            if grp["minutes"].sum() > 0
            else float(grp["rating_after"].mean())
        )
        rows.append({
            "season_start": int(s),
            "season_end": int(s) + 1,
            "n_exits": len(grp),
            "mean_exit": float(grp["rating_after"].mean()),
            "minutes_wmean_exit": wmean,
            "median_exit": float(grp["rating_after"].median()),
            "p25_exit": float(grp["rating_after"].quantile(0.25)),
            "p75_exit": float(grp["rating_after"].quantile(0.75)),
        })
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    print("loading usable boxscores 1976-2025", flush=True)
    box = load_usable_boxscores(1976, 2025)
    print(f"loaded {len(box)} rows, {box['game_id'].nunique()} games "
          f"({time.time() - t0:.1f}s)", flush=True)

    draft_ratings = build_draft_inputs(rookie_start=ROOKIE_START)
    engine = EloEngine(
        k=45.0,
        theta=0.3,
        home_advantage=70.0,
        alpha=1.0,
        rookie_boost=6.0,
        rookie_tau=60.0,
        init_mode="draft",
        draft_ratings=draft_ratings,
        rookie_ids=None,
        record_states=False,
        surprise_scale=285.0,
        surprise_anchor="game",
        playoff_k_multiplier=1.0,
        alpha_modern=1.0,
        alpha_modern_start_season=1996,
    )
    print("running engine", flush=True)
    t1 = time.time()
    updates, _, _ = engine.run(box)
    print(f"engine ran in {time.time() - t1:.1f}s: {len(updates)} updates", flush=True)

    final = final_ratings(updates)
    exits = exit_stats(final)
    RESULTS.mkdir(exist_ok=True)
    out_name = f"rookie_start_exit_calibration_{int(ROOKIE_START)}.csv" if len(sys.argv) > 1 else "rookie_start_exit_calibration.csv"
    exits.to_csv(RESULTS / out_name, index=False)

    per_player = exit_players(final)
    overall_wmean = float(
        np.average(
            per_player["rating_after"],
            weights=per_player["minutes"],
        )
    )
    print("per-season exit table:", flush=True)
    print(exits.to_string(index=False, float_format=lambda x: f"{x:.1f}"), flush=True)
    print(f"rookie_start={ROOKIE_START}", flush=True)
    print(f"overall minutes_wmean_exit={overall_wmean:.1f}", flush=True)
    print(f"entry minus exit = {ROOKIE_START - overall_wmean:+.1f}", flush=True)
    print(f"total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
