"""Build the R4/R6 resume checkpoint at the end of the 1995-96 season.

All R4 grid points share the same 1976-1995 history because theta and
alpha_modern are fixed there (theta=0, alpha=1).  This script runs that
shared segment once with the final locked parameters and saves the per-player
state (rating, games_played, games_played_track) so CV runs can resume
from season 1996 instead of re-running the full history.

The old theta=0.3 checkpoint (elo_checkpoint_1995.csv) is kept for
reproducibility; this script now writes elo_checkpoint_1995_theta0.csv.

Usage:
    python scripts/build_checkpoint.py [end_season]
"""

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_full_elo import build_draft_inputs, load_usable_boxscores
from src.elo.engine import EloEngine

RESULTS = Path("results")
END_SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 1995


def main():
    t0 = time.time()
    print(f"loading usable boxscores 1976-{END_SEASON}", flush=True)
    box = load_usable_boxscores(1976, END_SEASON)
    print(f"loaded {len(box)} rows, {box['game_id'].nunique()} games "
          f"({time.time() - t0:.1f}s)", flush=True)

    draft_ratings = build_draft_inputs(rookie_start=1425.0)
    engine = EloEngine(
        k=45.0,
        theta=0.0,
        home_advantage=70.0,
        alpha=1.0,
        rookie_boost=6.0,
        rookie_tau=60.0,
        init_mode="draft",
        draft_ratings=draft_ratings,
        rookie_ids=None,
        record_states=True,
        surprise_scale=285.0,
        surprise_anchor="game",
        playoff_k_multiplier=1.0,
        alpha_modern=1.0,
        alpha_modern_start_season=1996,
    )
    print("running engine", flush=True)
    t1 = time.time()
    updates, states, inits = engine.run(box)
    print(f"engine ran in {time.time() - t1:.1f}s: {len(updates)} updates, "
          f"{len(inits)} inits, {len(states)} state rows", flush=True)

    # Last state per player: rating and cumulative games played.
    last = (
        states.sort_values(["game_date", "game_id"])
        .groupby("player_id")
        .tail(1)[["player_id", "rating", "games_played", "games_played_track"]]
    )
    print(f"checkpoint players: {len(last)}", flush=True)
    out = RESULTS / f"elo_checkpoint_{END_SEASON}_theta0.csv"
    RESULTS.mkdir(exist_ok=True)
    last.to_csv(out, index=False)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    print(f"total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
