"""Run playoff_k variants for the delivery layer (D-025/D-036).

Three-segment protocol per new playoff_k value (pk=1.0 already exists in
results/final from E-031):

    shared     canonical params (scale=285, alpha=1) 1976-1995
               from the canonical bidirectional 1976 initial state
               -> variant 1995 checkpoint
    canonical  canonical params 1996-2025 from that checkpoint
    modern     modern params (scale=318, alpha=0.9) 1996-2025 from the
               same checkpoint

Outputs are organized under results/playoff_variants/:

    checkpoints/checkpoint_1995_pk<value>.csv
    shared/final_elo_updates_1976-1995_pk<value>.csv.gz
    canonical/final_elo_updates_canonical_pk<value>.csv.gz
    canonical/final_elo_diagnostics_canonical_pk<value>.csv
    modern/final_elo_updates_modern_pk<value>.csv.gz
    modern/final_elo_diagnostics_modern_pk<value>.csv
    run_summary.csv

Usage:
    python scripts/run_playoff_variants.py --values 0,0.5,1.5,2.0
    python scripts/run_playoff_variants.py --values 0,0.5,1.5,2.0 --force
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from run_final_elo import (
    CANONICAL,
    MODERN,
    checkpoint_from_updates,
    make_engine,
    write_checkpoint,
)
from run_full_elo import load_usable_boxscores, season_metrics

RESULTS = PROJECT_ROOT / "results"
VARIANTS = RESULTS / "playoff_variants"
CHECKPOINTS = VARIANTS / "checkpoints"
SHARED = VARIANTS / "shared"
CANON_DIR = VARIANTS / "canonical"
MODERN_DIR = VARIANTS / "modern"
INIT1976 = RESULTS / "final" / "final_elo_initial_state_1976_canonical.csv"
LOG_PATH = PROJECT_ROOT / "scratch" / "playoff_variants.log"


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def tag(value):
    return f"pk{value:g}"


def load_checkpoint_dict(path):
    df = pd.read_csv(path)
    return {
        int(r.player_id): (float(r.rating), int(r.games_played),
                           int(r.games_played_track))
        for r in df.itertuples(index=False)
    }


def run_shared(value, force):
    t0 = time.time()
    t = tag(value)
    out_updates = SHARED / f"final_elo_updates_1976-1995_{t}.csv.gz"
    out_ckpt = CHECKPOINTS / f"checkpoint_1995_{t}.csv"
    if out_updates.exists() and out_ckpt.exists() and not force:
        log(f"[shared pk={value}] outputs exist, skip")
        return {"value": value, "elapsed": 0.0, "updates": -1, "games": -1}
    box = load_usable_boxscores(1976, 1995)
    params = {**CANONICAL, "playoff_k": value}
    updates, _, _ = make_engine(
        params, initial_state=load_checkpoint_dict(INIT1976)
    ).run(box)
    updates.to_csv(out_updates, index=False, compression="gzip")
    write_checkpoint(out_ckpt, checkpoint_from_updates(updates, 1995))
    elapsed = time.time() - t0
    log(f"[shared pk={value}] {len(updates)} updates, "
        f"{updates['game_id'].nunique()} games in {elapsed:.1f}s")
    return {"value": value, "elapsed": round(elapsed, 1),
            "updates": int(len(updates)),
            "games": int(updates["game_id"].nunique())}


def run_continuation(track, value, force):
    t0 = time.time()
    t = tag(value)
    params = {**(CANONICAL if track == "canonical" else MODERN),
              "playoff_k": value}
    out_updates = VARIANTS / track / f"final_elo_updates_{track}_{t}.csv.gz"
    out_diag = VARIANTS / track / f"final_elo_diagnostics_{track}_{t}.csv"
    if out_updates.exists() and out_diag.exists() and not force:
        log(f"[{track} pk={value}] outputs exist, skip")
        return {"track": track, "value": value, "elapsed": 0.0,
                "updates": -1, "games": -1}
    ckpt_path = CHECKPOINTS / f"checkpoint_1995_{t}.csv"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing {ckpt_path}; run shared segments first")
    box = load_usable_boxscores(1996, 2025)
    updates, _, _ = make_engine(
        params, initial_state=load_checkpoint_dict(ckpt_path)
    ).run(box)
    updates.to_csv(out_updates, index=False, compression="gzip")
    season_metrics(updates, box).to_csv(out_diag, index=False)
    elapsed = time.time() - t0
    log(f"[{track} pk={value}] {len(updates)} updates, "
        f"{updates['game_id'].nunique()} games in {elapsed:.1f}s")
    return {"track": track, "value": value, "elapsed": round(elapsed, 1),
            "updates": int(len(updates)),
            "games": int(updates["game_id"].nunique())}


def write_summary(values):
    rows = []
    all_values = sorted(set(float(v) for v in values) | {1.0})
    for track in ("canonical", "modern"):
        for value in all_values:
            t = tag(value)
            if value == 1.0:
                updates_path = RESULTS / "final" / f"final_elo_updates_{track}.csv.gz"
                diag_path = RESULTS / "final" / f"final_elo_diagnostics_{track}.csv"
            else:
                updates_path = VARIANTS / track / f"final_elo_updates_{track}_{t}.csv.gz"
                diag_path = VARIANTS / track / f"final_elo_diagnostics_{track}_{t}.csv"
            row = {
                "track": track,
                "playoff_k": value,
                "updates_file": str(updates_path),
                "diagnostics_file": str(diag_path),
            }
            if value != 1.0:
                row["shared_file"] = str(
                    SHARED / f"final_elo_updates_1976-1995_{t}.csv.gz")
            if updates_path.exists():
                row["updates_bytes"] = int(updates_path.stat().st_size)
                row["n_updates"] = int(len(pd.read_csv(
                    updates_path, usecols=["game_id"], low_memory=False,
                    compression="gzip")))
                if track == "canonical" and value != 1.0:
                    shared_path = SHARED / f"final_elo_updates_1976-1995_{t}.csv.gz"
                    if shared_path.exists():
                        n_shared = int(len(pd.read_csv(
                            shared_path, usecols=["game_id"], low_memory=False,
                            compression="gzip")))
                        row["n_updates_total"] = row["n_updates"] + n_shared
            if diag_path.exists():
                diag = pd.read_csv(diag_path)
                last = diag[diag["season"] == diag["season"].max()].iloc[0]
                row["last_season"] = int(last["season"])
                row["last_sd"] = round(float(last["sd"]), 2)
                row["last_minutes_wmean"] = round(
                    float(last["minutes_wmean"]), 2)
                row["last_top_rating"] = round(float(last["top_rating"]), 2)
            rows.append(row)
    out = VARIANTS / "run_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"wrote {out} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", default="0.5,1.5,2.0",
                        help="comma-separated playoff_k values to run")
    parser.add_argument("--parallel-shared", type=int, default=3)
    parser.add_argument("--parallel-cont", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    values = [float(v) for v in args.values.split(",") if v.strip()]
    if not values:
        raise SystemExit("no playoff_k values given")
    for d in (VARIANTS, CHECKPOINTS, SHARED, CANON_DIR, MODERN_DIR):
        d.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    log(f"playoff variants start: values={values}, force={args.force}")
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.parallel_shared) as pool:
        futures = {pool.submit(run_shared, v, args.force): v for v in values}
        for fut in as_completed(futures):
            row = fut.result()
            log(f"shared done {row}")

    tasks = [(track, v) for track in ("canonical", "modern") for v in values]
    with ProcessPoolExecutor(max_workers=args.parallel_cont) as pool:
        futures = {
            pool.submit(run_continuation, track, v, args.force): (track, v)
            for track, v in tasks
        }
        for fut in as_completed(futures):
            row = fut.result()
            log(f"continuation done {row}")

    write_summary(values)
    log(f"all done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
