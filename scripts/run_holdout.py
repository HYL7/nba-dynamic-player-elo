"""Step 5 Hold-out evaluation (D-027/D-028): 2023-24 .. 2025-26.

The tuning grid never read seasons >= 2023, so the final dual-track run
(E-031) is evaluated on three untouched target seasons:

    boundary season s (end rating)   target season s+1 (first 10 games)
    2022-23 (2022)                   2023-24 (2023)
    2023-24 (2023)                   2024-25 (2024)
    2024-25 (2024)                   2025-26 (2025)

Four models are compared per track:

    L1  pure random prediction (target values permuted within boundary)
    L2  pure team Elo (team ratings from game results only, no player stats)
    L3  naive uniform parameters (pre-tuning default config)
    L4  locked optimal parameters (canonical 285/1.0, modern 318/0.9)

Outputs are the same four D-028 checks as the CV stats: Pearson/Spearman r,
MSE improvement over pred=0, cluster bootstrap 95% CI / p by player and by
boundary season, and the per-boundary exact binomial sign test.

Usage:
    python scripts/run_holdout.py                      # full run incl. L3 engine
    python scripts/run_holdout.py --no-l3              # skip the long L3 engine run
    python scripts/run_holdout.py --reuse-l3           # reuse stored L3 engine output
    python scripts/run_holdout.py --tracks canonical   # one track only
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from run_cv_elo import end_of_season_ratings, load_usable_boxscores, run_engine, target_perf
from cv_stats import (
    cluster_bootstrap,
    corr_p_normal,
    exact_binomial_tail,
    pearson_r,
    spearman_rho,
)
from src.elo.core import expected_win_probability

RESULTS = Path("results")
FINAL_DIR = RESULTS / "final"

TRACK_CONFIG = {
    "canonical": {"scale": 285.0, "alpha_ref": 1.0},
    "modern": {"scale": 318.0, "alpha_ref": 0.9},
}

NAIVE_PARAMS = {
    "k": 10.0,
    "scale": 400.0,
    "theta": 0.3,
    "h": 70.0,
    "alpha_modern": 1.0,
    "playoff_k": 1.0,
    "rookie_boost": 3.0,
    "rookie_tau": 25.0,
    "rookie_start": 1500.0,
    "on_court_mode": "raw",
}

UPDATES_COLS = [
    "season", "track", "game_date", "game_id", "player_id", "team_id",
    "rating_after", "z_game_score", "z_on_court_rate",
]
TEAM_COLS = ["season", "track", "game_date", "game_id", "team_id", "home", "s_team"]


def load_track_updates(track):
    path = FINAL_DIR / f"final_elo_updates_{track}.csv.gz"
    return pd.read_csv(path, usecols=UPDATES_COLS + ["home", "s_team"],
                       low_memory=False)


def build_elo_pairs(updates, scale, alpha_ref, boundary_start, boundary_end,
                    window=10, min_games=3):
    """D-027 pairs from end-of-season player Elo ratings."""
    ratings = end_of_season_ratings(updates)
    targets = target_perf(updates, window=window, min_games=min_games,
                          alpha_ref=alpha_ref)
    frames = []
    for season in range(boundary_start, boundary_end + 1):
        feat = ratings[ratings["season"] == season][
            ["player_id", "rating", "league_mean"]].copy()
        feat["pred"] = (feat["rating"] - feat["league_mean"]) / scale
        y = targets[targets["season"] == season + 1][["player_id", "target"]]
        joined = feat.merge(y, on="player_id", how="inner")
        joined["boundary_season"] = season
        frames.append(joined[["boundary_season", "player_id", "pred", "target"]])
    return pd.concat(frames, ignore_index=True)


def team_elo_ratings(updates, k=20.0, home_advantage=70.0):
    """Team ratings updated by game results only (pure team Elo, L2).

    One rating per franchise/team id, standard K*(S-E) update with the same
    home-advantage semantics as the player engine. Returns
    (season_team_rating, season_team_league_mean) at end of each season.
    """
    teams = (updates[TEAM_COLS].drop_duplicates()
             .sort_values(["game_date", "game_id", "team_id"]))
    ratings = {}
    season_end = {}
    for game_id, grp in teams.groupby("game_id", sort=False):
        rows = list(grp.itertuples(index=False))
        if len(rows) != 2:
            raise ValueError(f"team Elo game {game_id} has {len(rows)} teams")
        season = rows[0].season
        r = []
        for row in rows:
            tid = int(row.team_id)
            opp = [x for x in rows if int(x.team_id) != tid][0]
            diff = ratings.get(tid, 1500.0) - ratings.get(
                int(opp.team_id), 1500.0)
            if int(row.home) == 1:
                diff += home_advantage
            if int(opp.home) == 1:
                diff -= home_advantage
            e = expected_win_probability(diff)
            r.append((tid, float(row.s_team), e))
        for (tid, s, e), row in zip(r, rows):
            ratings[tid] = ratings.get(tid, 1500.0) + k * (s - e)
            season_end[(season, tid)] = ratings[tid]
    end = pd.DataFrame(
        [{"season": s, "team_id": t, "team_rating": v}
         for (s, t), v in season_end.items()]
    )
    league = end.groupby("season")["team_rating"].mean().rename("team_league_mean")
    return end.merge(league, on="season")


def build_team_pairs(updates, scale, alpha_ref, boundary_start, boundary_end,
                     window=10, min_games=3, team_k=20.0):
    """L2: each player inherits their last team's end-of-season rating."""
    end = team_elo_ratings(updates, k=team_k)
    targets = target_perf(updates, window=window, min_games=min_games,
                          alpha_ref=alpha_ref)
    last_team = (
        updates.sort_values(["game_date", "game_id"])
        .groupby(["season", "player_id"])["team_id"]
        .last()
        .rename("team_id")
        .reset_index()
    )
    frames = []
    for season in range(boundary_start, boundary_end + 1):
        feat = last_team[last_team["season"] == season].copy()
        feat = feat.merge(end[end["season"] == season],
                          on=["season", "team_id"], how="inner")
        feat["pred"] = (feat["team_rating"] - feat["team_league_mean"]) / scale
        y = targets[targets["season"] == season + 1][["player_id", "target"]]
        joined = feat.merge(y, on="player_id", how="inner")
        joined["boundary_season"] = season
        frames.append(joined[["boundary_season", "player_id", "pred", "target"]])
    return pd.concat(frames, ignore_index=True)


def random_pairs(pairs, seed=20260816):
    """L1: permute target values within each boundary season."""
    rng = np.random.default_rng(seed)
    groups = []
    for season, grp in pairs.groupby("boundary_season", sort=False):
        grp = grp.copy()
        y = grp["target"].to_numpy(dtype=float)
        grp["pred"] = rng.permutation(y)
        groups.append(grp)
    return pd.concat(groups, ignore_index=True)


def summarize(pairs, n_iter=2000, seed=20260816):
    """D-028 statistics for one model-track pair."""
    pred = pairs["pred"].to_numpy(dtype=float)
    target = pairs["target"].to_numpy(dtype=float)
    n = len(pairs)
    mse = float(np.mean((target - pred) ** 2))
    baseline_mse = float(np.mean(target ** 2))
    improvement = baseline_mse - mse
    pred_sd = float(pred.std(ddof=0))
    target_sd = float(target.std(ddof=0))

    r = pearson_r(pred, target)
    rho = spearman_rho(pred, target)
    p_r = corr_p_normal(r, n)

    per_boundary = {}
    for season, grp in pairs.groupby("boundary_season"):
        per_boundary[season] = pearson_r(grp["pred"], grp["target"])
    n_positive = int(sum(v > 0 for v in per_boundary.values()))
    p_sign = exact_binomial_tail(n_positive, len(per_boundary))

    corr_fn = lambda d: pearson_r(d["pred"], d["target"])
    gain_fn = lambda d: float(np.mean(
        d["target"] ** 2 - (d["target"] - d["pred"]) ** 2
    ))
    boot_player_corr = cluster_bootstrap(pairs, "player_id", corr_fn, n_iter, seed)
    boot_player_gain = cluster_bootstrap(pairs, "player_id", gain_fn, n_iter, seed)
    boot_season_corr = cluster_bootstrap(pairs, "boundary_season", corr_fn,
                                         n_iter, seed + 1)
    boot_season_gain = cluster_bootstrap(pairs, "boundary_season", gain_fn,
                                         n_iter, seed + 2)

    return {
        "n": n,
        "n_players": int(pairs["player_id"].nunique()),
        "n_boundaries": int(pairs["boundary_season"].nunique()),
        "mse": mse,
        "baseline_mse": baseline_mse,
        "improvement": improvement,
        "pred_sd": pred_sd,
        "target_sd": target_sd,
        "discrimination": pred_sd / target_sd if target_sd > 0 else np.nan,
        "r": r,
        "rho": rho,
        "p_r_normal": p_r,
        "n_positive_r": n_positive,
        "p_sign_exact": p_sign,
        "boot_player_r_mean": boot_player_corr[0],
        "boot_player_r_lo": boot_player_corr[1],
        "boot_player_r_hi": boot_player_corr[2],
        "boot_player_r_p": boot_player_corr[3],
        "boot_player_gain_mean": boot_player_gain[0],
        "boot_player_gain_lo": boot_player_gain[1],
        "boot_player_gain_hi": boot_player_gain[2],
        "boot_player_gain_p": boot_player_gain[3],
        "boot_season_r_mean": boot_season_corr[0],
        "boot_season_r_lo": boot_season_corr[1],
        "boot_season_r_hi": boot_season_corr[2],
        "boot_season_r_p": boot_season_corr[3],
        "boot_season_gain_mean": boot_season_gain[0],
        "boot_season_gain_lo": boot_season_gain[1],
        "boot_season_gain_hi": boot_season_gain[2],
        "boot_season_gain_p": boot_season_gain[3],
    }


def run_l3_engine():
    t0 = time.time()
    print("L3: loading usable boxscores 1976-2025", flush=True)
    box = load_usable_boxscores(1976, 2025)
    print(f"L3: loaded {len(box)} rows, {box['game_id'].nunique()} games "
          f"({time.time() - t0:.1f}s)", flush=True)
    t1 = time.time()
    updates = run_engine(box, NAIVE_PARAMS)
    print(f"L3: engine ran in {time.time() - t1:.1f}s, {len(updates)} updates",
          flush=True)
    cols = [c for c in UPDATES_COLS if c in updates.columns]
    out = updates[cols].copy()
    out.to_csv(RESULTS / "holdout_l3_updates.csv.gz", index=False,
               compression="gzip")
    print("L3: wrote results/holdout_l3_updates.csv.gz", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", nargs="+", default=["canonical", "modern"])
    parser.add_argument("--boundary-start", type=int, default=2022)
    parser.add_argument("--boundary-end", type=int, default=2024)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min-games", type=int, default=3)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260816)
    parser.add_argument("--no-l3", action="store_true")
    parser.add_argument("--reuse-l3", action="store_true",
                        help="reuse results/holdout_l3_updates.csv.gz from a prior run")
    parser.add_argument("--out", default="results/holdout")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    out_prefix = Path(args.out)
    all_pairs = []
    rows = []

    l3_updates = None
    if args.reuse_l3:
        l3_path = RESULTS / "holdout_l3_updates.csv.gz"
        if not l3_path.exists():
            raise SystemExit(f"{l3_path} not found; rerun without --reuse-l3")
        print(f"L3: reusing {l3_path}", flush=True)
        l3_updates = pd.read_csv(l3_path, low_memory=False)
    elif not args.no_l3:
        l3_updates = run_l3_engine()
        l3_path = RESULTS / "holdout_l3_updates.csv.gz"
        if not l3_path.exists():
            raise SystemExit(f"{l3_path} not found; rerun without --reuse-l3")
        print(f"L3: reusing {l3_path}", flush=True)
        l3_updates = pd.read_csv(l3_path, low_memory=False)

    for track in args.tracks:
        if track not in TRACK_CONFIG:
            raise SystemExit(f"unknown track: {track}")
        cfg = TRACK_CONFIG[track]
        print(f"[{track}] loading final updates", flush=True)
        t0 = time.time()
        updates = load_track_updates(track)
        print(f"[{track}] loaded {len(updates)} rows in "
              f"{time.time() - t0:.1f}s", flush=True)

        # L4 (locked optimal parameters) is evaluated from the final run.
        l4 = build_elo_pairs(updates, cfg["scale"], cfg["alpha_ref"],
                             args.boundary_start, args.boundary_end,
                             args.window, args.min_games)
        l1 = random_pairs(l4, seed=args.random_seed)
        l2 = build_team_pairs(updates, cfg["scale"], cfg["alpha_ref"],
                              args.boundary_start, args.boundary_end,
                              args.window, args.min_games)
        l3 = None
        if l3_updates is not None:
            l3 = build_elo_pairs(l3_updates, NAIVE_PARAMS["scale"], cfg["alpha_ref"],
                                 args.boundary_start, args.boundary_end,
                                 args.window, args.min_games)

        for model, pairs in [("L1", l1), ("L2", l2), ("L3", l3), ("L4", l4)]:
            if pairs is None or pairs.empty:
                continue
            pairs = pairs.copy()
            pairs["track"] = track
            pairs["model"] = model
            all_pairs.append(pairs)
            stat = summarize(pairs, n_iter=args.n_boot,
                             seed=args.random_seed)
            stat.update({"track": track, "model": model,
                         "scale": cfg["scale"], "alpha_ref": cfg["alpha_ref"],
                         "params": json.dumps(
                             NAIVE_PARAMS if model == "L3" else
                             {"k": 45.0, "theta": 0.0, "scale": cfg["scale"],
                              "alpha_modern": cfg["alpha_ref"]})})
            rows.append(stat)
            print(f"[{track}][{model}] n={stat['n']} MSE={stat['mse']:.4f} "
                  f"r={stat['r']:.4f} rho={stat['rho']:.4f} "
                  f"improvement={stat['improvement']:.4f} "
                  f"n_pos_r={stat['n_positive_r']}/{stat['n_boundaries']}",
                  flush=True)
        del updates
        gc.collect()

    if not all_pairs:
        raise SystemExit("no pairs produced")
    pairs_out = pd.concat(all_pairs, ignore_index=True)
    pairs_out.to_csv(f"{out_prefix}_pairs.csv.gz", index=False,
                     compression="gzip")
    pd.DataFrame(rows).to_csv(f"{out_prefix}_results.csv", index=False)
    print(f"wrote {out_prefix}_pairs.csv.gz ({len(pairs_out)} rows)")
    print(f"wrote {out_prefix}_results.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
