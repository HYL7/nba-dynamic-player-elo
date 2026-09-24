"""Final dual-track Elo runs: D-007 bidirectional + D-036 checkpoint resume.

Usage:
    python scripts/run_final_elo.py canonical [end_season]
    python scripts/run_final_elo.py modern [checkpoint_csv] [end_season]

canonical runs the locked 1.0 track through the full 3-pass bidirectional
protocol (forward warmup -> reverse propagation -> final forward) and writes
final updates, season diagnostics, the 1976-77 start ratings, and a 1995-96
end-of-season checkpoint used by the modern track.

modern resumes from that canonical checkpoint (default
results/final/final_elo_checkpoint_1995_canonical.csv) with the locked
0.9 track params (scale=318, alpha_modern=0.9) for 1996+.
"""

import gc
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_full_elo import build_draft_inputs, load_usable_boxscores, season_metrics
from src.elo.engine import EloEngine

RESULTS = Path("results")
FINAL_DIR = RESULTS / "final"

CANONICAL = {
    "k": 45.0,
    "scale": 285.0,
    "theta": 0.0,
    "h": 70.0,
    "alpha_modern": 1.0,
    "playoff_k": 1.0,
    "rookie_boost": 6.0,
    "rookie_tau": 60.0,
    "rookie_start": 1425.0,
    "on_court_mode": "raw",
}

MODERN = {
    **CANONICAL,
    "scale": 318.0,
    "alpha_modern": 0.9,
}


def make_engine(params, initial_state=None, reverse=False, record_states=False):
    draft_ratings = build_draft_inputs(rookie_start=float(params["rookie_start"]))
    return EloEngine(
        k=float(params["k"]),
        theta=float(params["theta"]),
        home_advantage=float(params["h"]),
        alpha=1.0,
        rookie_boost=float(params["rookie_boost"]),
        rookie_tau=float(params["rookie_tau"]),
        init_mode="draft",
        draft_ratings=draft_ratings,
        rookie_ids=None,
        record_states=record_states,
        surprise_scale=float(params["scale"]),
        surprise_anchor="game",
        playoff_k_multiplier=float(params["playoff_k"]),
        alpha_modern=float(params["alpha_modern"]),
        alpha_modern_start_season=1996,
        initial_state=initial_state,
        on_court_mode=params["on_court_mode"],
        reverse=reverse,
    )


def last_state_map(states):
    """player_id -> (rating, games_played, games_played_track) at window end."""
    last = (
        states.sort_values(["game_date", "game_id"])
        .groupby("player_id")
        .tail(1)
    )
    return {
        int(r.player_id): (float(r.rating), int(r.games_played), int(r.games_played_track))
        for r in last.itertuples(index=False)
    }


def last_rating_map(updates):
    """player_id -> last rating_after in window order."""
    last = (
        updates.sort_values(["game_date", "game_id"])
        .groupby("player_id")
        .tail(1)
    )
    return {int(r.player_id): float(r.rating_after) for r in last.itertuples(index=False)}


def checkpoint_from_updates(updates, upto_season):
    """Reconstruct engine state at the end of `upto_season` from updates."""
    sub = updates[updates["season"] <= upto_season].sort_values(["game_date", "game_id"])
    last = sub.groupby("player_id").tail(1)
    counts = sub.groupby("player_id").size()
    return {
        int(r.player_id): (
            float(r.rating_after), int(counts[int(r.player_id)]), int(counts[int(r.player_id)])
        )
        for r in last.itertuples(index=False)
    }


def write_checkpoint(path, state):
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {"player_id": pid, "rating": rating, "games_played": played, "games_played_track": track_played}
        for pid, (rating, played, track_played) in sorted(state.items())
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path} ({len(rows)} players)")


def save_outputs(track, updates, box):
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    updates_out = FINAL_DIR / f"final_elo_updates_{track}.csv.gz"
    updates.to_csv(updates_out, index=False, compression="gzip")
    diag = season_metrics(updates, box)
    diag_out = FINAL_DIR / f"final_elo_diagnostics_{track}.csv"
    diag.to_csv(diag_out, index=False)
    print(f"wrote {updates_out} ({updates_out.stat().st_size / 1e6:.0f} MB)")
    print(f"wrote {diag_out}")
    return diag


def run_canonical(end_season):
    t0 = time.time()
    print(f"loading usable boxscores 1976-{end_season}")
    box = load_usable_boxscores(1976, end_season)
    print(f"loaded {len(box)} rows, {box['game_id'].nunique()} games ({time.time() - t0:.1f}s)")

    t1 = time.time()
    print("pass 1: forward warmup", flush=True)
    up1, states1, _ = make_engine(CANONICAL, record_states=True).run(box)
    state1 = last_state_map(states1)
    print(f"pass 1 done in {time.time() - t1:.1f}s: {len(state1)} players", flush=True)
    del up1, states1
    gc.collect()

    t2 = time.time()
    print("pass 2: reverse propagation", flush=True)
    up2, _, _ = make_engine(CANONICAL, initial_state=state1, reverse=True).run(box)
    ratings2 = last_rating_map(up2)
    print(f"pass 2 done in {time.time() - t2:.1f}s", flush=True)
    del state1, up2
    gc.collect()

    first_season = int(box["season"].min())
    first_pids = set(box.loc[box["season"] == first_season, "personId"].astype(int))
    init1976 = {
        pid: (ratings2[pid], 0, 0) for pid in first_pids if pid in ratings2
    }
    init_out = FINAL_DIR / f"final_elo_initial_state_{first_season}_canonical.csv"
    write_checkpoint(init_out, init1976)
    del ratings2
    gc.collect()

    t3 = time.time()
    print("pass 3: final forward", flush=True)
    up3, _, _ = make_engine(CANONICAL, initial_state=init1976).run(box)
    print(f"pass 3 done in {time.time() - t3:.1f}s", flush=True)

    save_outputs("canonical", up3, box)
    ckpt = checkpoint_from_updates(up3, 1995)
    write_checkpoint(FINAL_DIR / "final_elo_checkpoint_1995_canonical.csv", ckpt)
    print(f"total {time.time() - t0:.1f}s")


def run_modern(checkpoint_csv, end_season):
    t0 = time.time()
    ckpt = pd.read_csv(checkpoint_csv)
    initial = {
        int(r.player_id): (float(r.rating), int(r.games_played), int(r.games_played_track))
        for r in ckpt.itertuples(index=False)
    }
    print(f"loaded checkpoint {checkpoint_csv} ({len(initial)} players)")

    print(f"loading usable boxscores 1996-{end_season}")
    box = load_usable_boxscores(1996, end_season)
    print(f"loaded {len(box)} rows, {box['game_id'].nunique()} games ({time.time() - t0:.1f}s)")

    t1 = time.time()
    updates, _, _ = make_engine(MODERN, initial_state=initial).run(box)
    print(f"modern track ran in {time.time() - t1:.1f}s: {len(updates)} updates")

    save_outputs("modern", updates, box)
    ckpt = checkpoint_from_updates(updates, end_season)
    write_checkpoint(FINAL_DIR / f"final_elo_checkpoint_{end_season}_modern.csv", ckpt)
    print(f"total {time.time() - t0:.1f}s")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "canonical"
    if mode == "canonical":
        end_season = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
        run_canonical(end_season)
    elif mode == "modern":
        checkpoint_csv = (
            sys.argv[2]
            if len(sys.argv) > 2
            else str(FINAL_DIR / "final_elo_checkpoint_1995_canonical.csv")
        )
        end_season = int(sys.argv[3]) if len(sys.argv) > 3 else 2025
        run_modern(checkpoint_csv, end_season)
    else:
        raise SystemExit("mode must be 'canonical' or 'modern'")


if __name__ == "__main__":
    main()
