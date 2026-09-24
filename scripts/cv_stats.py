"""Statistical checks for Elo predictive power (D-028).

Reuses the stored full-window updates parquet so the checks run without an
engine rerun. The parquet was produced with the E-004 baseline parameters
(K=10, scale=400, anchor=game, alpha=1), which matches the CV baseline.

Checks:
1. Pearson / Spearman correlation between pred and target, with a naive
   normal-approximation p-value plus cluster bootstrap (by player, by season).
2. Paired MSE improvement over the pred=0 baseline, with cluster bootstrap.
3. Per-boundary sign consistency (exact binomial test).

Usage:
    python cv_stats.py
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from run_cv_elo import end_of_season_ratings, target_perf

RESULTS = Path("results")


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman_rho(x, y):
    rx = pd.Series(x).rank().to_numpy(dtype=float)
    ry = pd.Series(y).rank().to_numpy(dtype=float)
    return pearson_r(rx, ry)


def corr_p_normal(r, n):
    """Two-sided p-value for r=0 using the normal approximation of t."""
    if n <= 2 or abs(r) >= 1.0:
        return np.nan
    t = r * math.sqrt((n - 2) / (1.0 - r * r))
    return 2.0 * (1.0 - norm_cdf(abs(t)))


def cluster_bootstrap(data, cluster_col, stat_fn, n_iter=2000, seed=20260814):
    clusters = data[cluster_col].to_numpy()
    unique = np.unique(clusters)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_iter)
    for k in range(n_iter):
        ids = rng.choice(unique, size=len(unique), replace=True)
        mask = np.isin(clusters, ids)
        sample = data.loc[mask]
        estimates[k] = stat_fn(sample)
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    # Bootstrap p-value for H0: statistic <= 0 against H1: statistic > 0.
    p_value = float((estimates <= 0.0).mean())
    return float(estimates.mean()), float(lo), float(hi), p_value


def build_pairs(updates, boundary_start, boundary_end, scale, window, min_games):
    end = end_of_season_ratings(updates)
    targets = target_perf(updates, window=window, min_games=min_games)
    frames = []
    for season in range(boundary_start, boundary_end + 1):
        feat = end[end["season"] == season][
            ["player_id", "rating", "league_mean"]
        ].copy()
        feat["pred"] = (feat["rating"] - feat["league_mean"]) / scale
        y = targets[targets["season"] == season + 1][
            ["player_id", "target", "count"]
        ]
        joined = feat.merge(y, on="player_id", how="inner")
        joined["boundary_season"] = season
        frames.append(joined)
    return pd.concat(frames, ignore_index=True)


def exact_binomial_tail(k, n):
    return sum(math.comb(n, i) * 0.5 ** n for i in range(k, n + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary-start", type=int, default=1997)
    parser.add_argument("--boundary-end", type=int, default=2021)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min-games", type=int, default=3)
    parser.add_argument("--scale", type=float, default=400.0)
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    updates = pd.read_parquet(RESULTS / "full_elo_updates_1976-2025.parquet")
    pairs = build_pairs(
        updates,
        boundary_start=args.boundary_start,
        boundary_end=args.boundary_end,
        scale=args.scale,
        window=args.window,
        min_games=args.min_games,
    )
    pairs.to_parquet(RESULTS / f"cv_pairs_{args.boundary_start}-{args.boundary_end}.parquet", index=False)

    pred = pairs["pred"].to_numpy(dtype=float)
    target = pairs["target"].to_numpy(dtype=float)
    n = len(pairs)
    model_mse = float(np.mean((target - pred) ** 2))
    baseline_mse = float(np.mean(target ** 2))
    improvement = baseline_mse - model_mse
    per_obs_gain = float(np.mean(target ** 2 - (target - pred) ** 2))
    sd_gain = float(np.std(target ** 2 - (target - pred) ** 2, ddof=1))
    t_gain = per_obs_gain / (sd_gain / math.sqrt(n)) if sd_gain > 0 else np.nan
    p_gain_normal = 1.0 - norm_cdf(t_gain)

    r = pearson_r(pred, target)
    rho = spearman_rho(pred, target)
    p_r = corr_p_normal(r, n)

    per_boundary = {}
    for season, grp in pairs.groupby("boundary_season"):
        per_boundary[season] = pearson_r(grp["pred"], grp["target"])
    n_positive = int(sum(v > 0 for v in per_boundary.values()))
    p_sign = exact_binomial_tail(n_positive, len(per_boundary))

    n_players = pairs["player_id"].nunique()
    n_seasons = pairs["boundary_season"].nunique()

    corr_fn = lambda d: pearson_r(d["pred"], d["target"])
    gain_fn = lambda d: float(np.mean(
        d["target"] ** 2 - (d["target"] - d["pred"]) ** 2
    ))
    boot_player_corr = cluster_bootstrap(
        pairs, "player_id", corr_fn, args.n_iter, args.seed
    )
    boot_player_gain = cluster_bootstrap(
        pairs, "player_id", gain_fn, args.n_iter, args.seed
    )
    boot_season_corr = cluster_bootstrap(
        pairs, "boundary_season", corr_fn, args.n_iter, args.seed + 1
    )
    boot_season_gain = cluster_bootstrap(
        pairs, "boundary_season", gain_fn, args.n_iter, args.seed + 2
    )

    print(f"baseline: n={n}, players={n_players}, boundaries={n_seasons}")
    print(f"model_mse={model_mse:.5f}, baseline_mse={baseline_mse:.5f}, "
          f"improvement={improvement:.5f}")
    pred_sd = float(pred.std(ddof=0))
    target_sd = float(target.std(ddof=0))
    print(f"pred_sd={pred_sd:.4f}, target_sd={target_sd:.4f}, "
          f"discrimination={pred_sd / target_sd if target_sd > 0 else float('nan'):.3f}")
    print(f"pearson r={r:.4f} (naive p={p_r:.3g}), "
          f"spearman rho={rho:.4f}")
    print(f"per-boundary positive r: {n_positive}/{n_seasons} "
          f"(exact binomial p={p_sign:.3g})")
    print(f"per-obs gain={per_obs_gain:.4f}, naive t p={p_gain_normal:.3g}")
    print("bootstrap by player (2000):")
    print(f"  r mean={boot_player_corr[0]:.4f} "
          f"95% CI=({boot_player_corr[1]:.4f}, {boot_player_corr[2]:.4f}) "
          f"p={boot_player_corr[3]:.3g}")
    print(f"  gain mean={boot_player_gain[0]:.4f} "
          f"95% CI=({boot_player_gain[1]:.4f}, {boot_player_gain[2]:.4f}) "
          f"p={boot_player_gain[3]:.3g}")
    print("bootstrap by season (2000):")
    print(f"  r mean={boot_season_corr[0]:.4f} "
          f"95% CI=({boot_season_corr[1]:.4f}, {boot_season_corr[2]:.4f}) "
          f"p={boot_season_corr[3]:.3g}")
    print(f"  gain mean={boot_season_gain[0]:.4f} "
          f"95% CI=({boot_season_gain[1]:.4f}, {boot_season_gain[2]:.4f}) "
          f"p={boot_season_gain[3]:.3g}")


if __name__ == "__main__":
    main()
