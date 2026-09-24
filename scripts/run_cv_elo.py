"""Rolling time-series CV for DESIGN 3.4 Step 3 parameter search.

The sequential engine only reads past games, so one forward pass per
parameter set yields valid end-of-season states for every boundary season.

Prediction target (D-027):
    boundary season t: 1997-98 .. 2021-22 (internal years 1997..2021)
    rating_i      = last rating_after of season t
    league_mean   = mean of those end ratings
    predicted perf_i = (rating_i - league_mean) / surprise_scale
    target_i      = mean alpha_ref-blended label over the player's first
                    `window` regular-season games of season t+1 (>= min_games)
                    (D-027a baseline is alpha_ref=1.0 = pure z_game_score;
                    E-025 runs alpha_ref = alpha_modern so the sensitivity
                    grid moves Elo updates and the CV target together)
    targets are 1998-99 .. 2022-23 (internal years 1998..2022), so the
    2023-24 hold-out stays untouched.
    loss          = unweighted MSE over all player-boundaries

Every tunable engine parameter (D-030) has a CLI flag and can be gridded:
k, scale, theta, h, alpha_modern, rookie_boost, rookie_tau, rookie_start, on_court_mode.
playoff_k is also accepted but is deferred to the final playoff experiment (D-025);
alpha_ref sets the CV label blend (default: equals alpha_modern, E-025).

DEFAULT_PARAMS reflects the locked 1.0 canonical track; grids override them.

Discrimination diagnostics (D-031): pred_sd / target_sd is reported for
each run so MSE gains cannot hide a compressed, low-spread rating scale.

Win diagnostics (R5 auxiliary, D-005): per-game log loss and Brier score of
E_team vs the realized S_team. E_team is computed from pre-game ratings,
so the metrics are out-of-sample and show whether team results carry any
predictive value beyond the individual-performance Elo.

Team-win diagnostics (E-025 auxiliary): mean in-season E_team vs realized
regular-season win rate per team-season (Pearson r) is written to
cv_results.csv; --save-snapshots additionally writes season-end, All-Star
cutoff and top-player snapshots per parameter set.

Usage:
    python scripts/run_cv_elo.py --k 45 --scale 285 --theta 0 --h 70 --alpha-modern 1
    python scripts/run_cv_elo.py --grid grids/coarse_k_scale.json
    python scripts/run_cv_elo.py --grid grids/all_params.json --parallel 6
"""

import argparse
import itertools
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from run_full_elo import build_draft_inputs, load_usable_boxscores, season_metrics
from src.elo.engine import EloEngine

RESULTS = Path("results")

DEFAULT_PARAMS = {
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


def run_engine(box, params, initial_state=None):
    draft_ratings = build_draft_inputs(rookie_start=float(params["rookie_start"]))
    engine = EloEngine(
        k=float(params["k"]),
        theta=float(params["theta"]),
        home_advantage=float(params["h"]),
        alpha=1.0,
        rookie_boost=float(params["rookie_boost"]),
        rookie_tau=float(params["rookie_tau"]),
        init_mode="draft",
        draft_ratings=draft_ratings,
        rookie_ids=None,
        record_states=False,
        surprise_scale=float(params["scale"]),
        surprise_anchor="game",
        playoff_k_multiplier=float(params.get("playoff_k", 1.0)),
        alpha_modern=float(params["alpha_modern"]),
        alpha_modern_start_season=1996,
        initial_state=initial_state,
        on_court_mode=params.get("on_court_mode", "raw"),
    )
    updates, _, _ = engine.run(box)
    return updates


def end_of_season_ratings(updates):
    """Last rating_after per player per season, plus the season league mean."""
    rows = (
        updates.sort_values(["game_date", "game_id"])
        .groupby(["season", "player_id"])["rating_after"]
        .last()
        .rename("rating")
        .reset_index()
    )
    league_mean = rows.groupby("season")["rating"].mean().rename("league_mean")
    return rows.merge(league_mean, on="season")

def ratings_as_of_team_game(updates, game_n):
    """Last rating per player once each team has played `game_n` regular-season games.

    R7 sensitivity snapshot: use the regular-season schedule per (season,
    team_id), keep every player whose team reached the cutoff, and take each
    player's latest rating up to that point. Seasons shorter than game_n
    (e.g. the 1998-99 lockout at 50 games) fall back to the last regular
    game so the boundary keeps coverage. Playoff games are ignored.
    """
    reg = updates[updates["track"] == "regular"].copy()
    if reg.empty:
        return pd.DataFrame(columns=["season", "player_id", "rating", "league_mean"])
    games = (
        reg[["season", "team_id", "game_id", "game_date"]]
        .drop_duplicates()
        .sort_values(["season", "team_id", "game_date", "game_id"])
    )
    games["team_game_no"] = games.groupby(["season", "team_id"]).cumcount() + 1
    max_no = games.groupby("season")["team_game_no"].max().rename("max_no")
    games = games.merge(max_no, on="season")
    games["cutoff"] = np.minimum(game_n, games["max_no"]).astype(int)
    eligible = games[games["team_game_no"] <= games["cutoff"]]
    eligible = eligible[["season", "team_id", "game_id"]]
    snap = reg.merge(eligible, on=["season", "team_id", "game_id"], how="inner")
    last = (
        snap.sort_values(["game_date", "game_id"])
        .groupby(["season", "player_id"])
        .tail(1)
    )
    out = last[["season", "player_id", "rating_after"]].rename(
        columns={"rating_after": "rating"})
    league_mean = out.groupby("season")["rating"].mean().rename("league_mean")
    return out.merge(league_mean, on="season")

def next_window_after_team_games(updates, game_n, window=10, min_games=3,
                                 alpha_ref=1.0):
    """Mean alpha_ref label over the next `window` games after the team-game cutoff.

    R7 corrected protocol: a season-N sampling snapshot covers the first N
    regular-season games of each team, and each player's target is the mean
    label over their next `window` regular-season appearances strictly after
    their own snapshot point in the same season. Playoff games are ignored.
    Seasons shorter than game_n fall back to the last regular game, so no
    target remains for that boundary (consistent with ratings_as_of_team_game).
    """
    reg = updates[updates["track"] == "regular"].copy()
    if reg.empty:
        return pd.DataFrame(columns=["season", "player_id", "target", "count"])
    games = (
        reg[["season", "team_id", "game_id", "game_date"]]
        .drop_duplicates()
        .sort_values(["season", "team_id", "game_date", "game_id"])
    )
    games["team_game_no"] = games.groupby(["season", "team_id"]).cumcount() + 1
    max_no = games.groupby("season")["team_game_no"].max().rename("max_no")
    games = games.merge(max_no, on="season")
    games["cutoff"] = np.minimum(game_n, games["max_no"]).astype(int)
    eligible = games[games["team_game_no"] <= games["cutoff"]]
    eligible = eligible[["season", "team_id", "game_id"]]
    snap = reg.merge(eligible, on=["season", "team_id", "game_id"], how="inner")
    boundary = (
        snap.sort_values(["game_date", "game_id"])
        .groupby(["season", "player_id"])
        .tail(1)
        [["season", "player_id", "game_date", "game_id"]]
        .rename(columns={"game_date": "cut_date", "game_id": "cut_game"})
    )
    reg = reg.merge(boundary, on=["season", "player_id"], how="inner")
    after = (reg["game_date"] > reg["cut_date"]) | (
        (reg["game_date"] == reg["cut_date"]) & (reg["game_id"] > reg["cut_game"]))
    reg = reg[after].sort_values(["season", "player_id", "game_date", "game_id"])
    if reg.empty:
        return pd.DataFrame(columns=["season", "player_id", "target", "count"])
    reg["label"] = blended_label(reg, alpha_ref)
    first_rows = reg.groupby(["season", "player_id"]).head(window)
    agg = (
        first_rows.groupby(["season", "player_id"])["label"]
        .agg(["mean", "count"])
        .reset_index()
    )
    return agg[agg["count"] >= min_games].rename(columns={"mean": "target"})


def blended_label(updates, alpha_ref=1.0):
    """Build the alpha_ref CV label from stored z-score columns.

    Mirrors compute_game_performance's fallback: a game with no on-court
    signal at all keeps pure z_game_score (pre-1996 behavior); in mixed
    games, rows without on-court data contribute 0 to the on-court term.
    """
    gs = updates["z_game_score"].to_numpy(dtype=float)
    if "z_on_court_rate" not in updates.columns or alpha_ref >= 1.0:
        return gs
    oc = updates["z_on_court_rate"]
    game_has_oc = oc.notna().groupby(updates["game_id"]).transform("any")
    label = alpha_ref * gs + (1.0 - alpha_ref) * oc.fillna(0.0).to_numpy(dtype=float)
    return np.where(game_has_oc, label, gs)


def target_perf(updates, window=10, min_games=3, alpha_ref=1.0):
    """Mean alpha_ref label over the first `window` games (D-027b).

    alpha_ref defaults to 1.0 (pure GameScore z, the D-027a baseline).
    E-025 sets it equal to alpha_modern so the sensitivity grid moves the
    Elo updates and the CV target together.
    """
    regular = updates[updates["track"] == "regular"]
    if regular.empty:
        return pd.DataFrame(columns=["season", "player_id", "target", "count"])
    first_rows = (
        regular.sort_values(["game_date", "game_id"])
        .groupby(["season", "player_id"])
        .head(window)
        .copy()
    )
    first_rows["label"] = blended_label(first_rows, alpha_ref)
    agg = (
        first_rows.groupby(["season", "player_id"])["label"]
        .agg(["mean", "count"])
        .reset_index()
    )
    return agg[agg["count"] >= min_games].rename(columns={"mean": "target"})

def win_prediction_metrics(updates):
    """Out-of-sample win log loss and Brier from stored s_team/e_team rows.

    Each game has one row per player, so s_team/e_team are de-duplicated
    per (game_id, team_id) before averaging the two team predictions per
    game. Returns NaN when the updates frame lacks the columns (synthetic
    unit-test frames).
    """
    required = {"game_id", "team_id", "s_team", "e_team"}
    if not required.issubset(updates.columns):
        return {"win_log_loss": np.nan, "win_brier": np.nan, "win_games": 0}
    teams = (
        updates.sort_values(["game_date", "game_id", "player_id"])
        .groupby(["game_id", "team_id"])[["s_team", "e_team"]]
        .first()
        .reset_index()
    )
    e = teams["e_team"].clip(1e-9, 1.0 - 1e-9)
    log_loss = -(teams["s_team"] * np.log(e)
                 + (1.0 - teams["s_team"]) * np.log(1.0 - e)).mean()
    brier = ((teams["s_team"] - teams["e_team"]) ** 2).mean()
    return {
        "win_log_loss": float(log_loss),
        "win_brier": float(brier),
        "win_games": len(teams) // 2,
    }

def team_win_metrics(updates, min_season=1996):
    """Mean in-season E_team vs realized regular-season win rate (E-025).

    Each game's e_team comes from that game's actual players' pre-game
    ratings, so mid-season trades are handled naturally: the rating follows
    the player and the team expectation reflects who is really on the floor.
    Correlating the season mean of e_team with the realized regular-season
    win rate avoids the roster-continuity problem of comparing one season's
    end-of-season lineup with another season's record. Returns NaN/0 when
    the updates frame lacks the needed columns (synthetic unit-test frames).
    """
    required = {"game_id", "game_date", "season", "track", "team_id",
                "s_team", "e_team"}
    empty = {"team_win_corr": np.nan, "team_win_pairs": 0, "team_win_games": 0}
    if not required.issubset(updates.columns):
        return dict(empty)
    reg = updates[updates["track"] == "regular"]
    reg = reg[reg["season"] >= min_season]
    if reg.empty:
        return dict(empty)
    teams = (
        reg.sort_values(["game_date", "game_id"])
        .groupby(["game_id", "team_id"], sort=False)[["s_team", "e_team"]]
        .first()
        .reset_index()
    )
    seasons = reg[["game_id", "team_id", "season"]].drop_duplicates()
    teams = teams.merge(seasons, on=["game_id", "team_id"], how="left")
    team_season = (
        teams.groupby(["season", "team_id"], sort=False)
        .agg(exp_win_rate=("e_team", "mean"),
             win_rate=("s_team", "mean"),
             n_games=("game_id", "nunique"))
        .reset_index()
    )
    if len(team_season) < 2:
        return {"team_win_corr": np.nan, "team_win_pairs": len(team_season),
                "team_win_games": 0}
    corr = np.corrcoef(team_season["exp_win_rate"], team_season["win_rate"])[0, 1]
    return {
        "team_win_corr": float(corr),
        "team_win_pairs": int(len(team_season)),
        "team_win_games": int(team_season["n_games"].sum()),
    }


def all_star_cutoff(season):
    """Approximate All-Star roster snapshot date for an internal season."""
    if season == 1998:
        return "1999-02-14"
    if season == 2020:
        return "2021-03-01"
    return f"{season + 1}-02-01"


def snapshot_at(updates, season, cutoff):
    """Latest rating per player before an All-Star cutoff in one season."""
    frame = updates[
        (updates["season"] == season)
        & (updates["game_date"].astype(str) <= str(cutoff))
    ]
    if frame.empty:
        return pd.DataFrame(
            columns=["season", "season_end", "rank", "player_id",
                     "rating_after", "games_played"]
        )
    last = frame.sort_values(["game_date", "game_id"]).groupby("player_id").tail(1)
    games = frame.groupby("player_id")["game_id"].nunique().rename("games_played")
    out = last[["player_id", "rating_after"]].merge(
        games, left_on="player_id", right_index=True
    )
    out["season"] = int(season)
    out["season_end"] = int(season) + 1
    out["rank"] = out["rating_after"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("rank")[
        ["season", "season_end", "rank", "player_id", "rating_after", "games_played"]
    ].reset_index(drop=True)


def top_players_snapshot(updates, seasons=(1997, 2007, 2017, 2021), top_n=10):
    """End-of-season top-N ratings for selected display seasons."""
    rows = []
    for season in seasons:
        sub = updates[updates["season"] == season]
        if sub.empty:
            continue
        end = end_of_season_ratings(sub)
        top = end.nlargest(top_n, "rating").copy()
        top["season_end"] = int(season) + 1
        top["rank"] = np.arange(1, len(top) + 1)
        rows.append(top[["season", "season_end", "rank", "player_id",
                         "rating", "league_mean"]])
    if not rows:
        return pd.DataFrame(columns=["season", "season_end", "rank", "player_id",
                                     "rating", "league_mean"])
    return pd.concat(rows, ignore_index=True)


def load_player_names():
    """personId -> full name lookup for human-readable snapshots."""
    try:
        players = pd.read_csv(
            "clean_data/players_clean.csv",
            usecols=["personId", "firstName", "lastName"],
            low_memory=False,
        )
    except FileNotFoundError:
        return pd.DataFrame(columns=["player_id", "player_name"])
    name = (
        players["firstName"].fillna("") + " " + players["lastName"].fillna("")
    ).str.strip()
    return pd.DataFrame(
        {"player_id": players["personId"].astype(int), "player_name": name}
    )


def save_snapshots(updates, merged, cfg, names=None):
    """Write season-end, All-Star and top-player snapshots for one run."""
    tag = (f"alpha_ref_{float(merged['alpha_ref']):g}"
           f"_oc_{merged.get('on_court_mode', 'raw')}"
           f"_k{float(merged['k']):g}_scale{float(merged['scale']):g}")
    RESULTS.mkdir(exist_ok=True)

    end = end_of_season_ratings(updates)
    end["season_end"] = end["season"] + 1
    if names is not None and not names.empty:
        end = end.merge(names, on="player_id", how="left")
    end.to_csv(RESULTS / f"{tag}_season_end.csv", index=False)

    allstar = [
        snapshot_at(updates, int(season), all_star_cutoff(int(season)))
        for season in sorted(updates["season"].unique())
    ]
    allstar = [s for s in allstar if not s.empty]
    if allstar:
        snap = pd.concat(allstar, ignore_index=True)
        if names is not None and not names.empty:
            snap = snap.merge(names, on="player_id", how="left")
        snap.to_csv(RESULTS / f"{tag}_allstar_snapshot.csv", index=False)

    top = top_players_snapshot(updates)
    if names is not None and not names.empty:
        top = top.merge(names, on="player_id", how="left")
    top.to_csv(RESULTS / f"{tag}_top_players.csv", index=False)


def cv_metric(updates, params, boundary_start, boundary_end,
              window=10, min_games=3, sampling="season_end", verbose=False):
    alpha_ref = float(params.get("alpha_ref", 1.0))
    scale = float(params["scale"])
    if sampling == "season_end":
        ratings = end_of_season_ratings(updates)
        targets = target_perf(updates, window=window, min_games=min_games,
                              alpha_ref=alpha_ref)
        target_season_offset = 1
    elif sampling.startswith("game"):
        game_n = int(sampling[4:])
        ratings = ratings_as_of_team_game(updates, game_n)
        targets = next_window_after_team_games(
            updates, game_n, window=window, min_games=min_games,
            alpha_ref=alpha_ref)
        target_season_offset = 0
    else:
        raise ValueError(f"unknown sampling: {sampling}")
    scale = float(params["scale"])

    pairs = []
    boundaries = list(range(boundary_start, boundary_end + 1))
    for idx, season in enumerate(boundaries, 1):
        if verbose and (idx % 5 == 0 or idx == len(boundaries)):
            print(f"  metric boundary {season}/{boundaries[-1]} ({idx}/{len(boundaries)})", flush=True)
        feat = ratings[ratings["season"] == season][
            ["player_id", "rating", "league_mean"]
        ].copy()
        feat["pred"] = (feat["rating"] - feat["league_mean"]) / scale
        y = targets[targets["season"] == season + target_season_offset][
            ["player_id", "target"]]
        joined = feat.merge(y, on="player_id", how="inner")
        if not joined.empty:
            pairs.append(joined[["pred", "target"]])

    if not pairs:
        metric = {"mse": np.nan, "r": np.nan, "baseline_mse": np.nan,
                 "pred_sd": np.nan, "target_sd": np.nan,
                 "discrimination": np.nan, "n_players": 0, "n_boundaries": 0}
        metric.update(win_prediction_metrics(updates))
        metric.update(team_win_metrics(updates))
        return metric

    pair_df = pd.concat(pairs, ignore_index=True)
    err = pair_df["target"] - pair_df["pred"]
    corr = np.corrcoef(pair_df["pred"], pair_df["target"])[0, 1]
    pred_sd = float(pair_df["pred"].std(ddof=0))
    target_sd = float(pair_df["target"].std(ddof=0))
    metric = {
        "mse": float(np.mean(err**2)),
        "r": float(corr),
        "baseline_mse": float(np.mean(pair_df["target"] ** 2)),
        "pred_sd": pred_sd,
        "target_sd": target_sd,
        "discrimination": pred_sd / target_sd if target_sd > 0 else np.nan,
        "n_players": len(pair_df),
        "n_boundaries": len(pairs),
    }
    metric.update(win_prediction_metrics(updates))
    metric.update(team_win_metrics(updates))
    return metric


def build_row(updates, merged, box, cfg, elapsed, sampling="season_end"):
    metric = cv_metric(
        updates, merged,
        boundary_start=cfg["boundary_start"],
        boundary_end=cfg["boundary_end"],
        window=cfg["window"],
        min_games=cfg["min_games"],
        sampling=sampling,
        verbose=cfg.get("verbose", False),
    )
    scale_rows = season_metrics(updates, box)
    last_scale = scale_rows[scale_rows["season"] == scale_rows["season"].max()].iloc[0]
    return {
        "sampling": sampling,
        "k": merged["k"],
        "scale": merged["scale"],
        "theta": merged["theta"],
        "h": merged["h"],
        "alpha_modern": merged["alpha_modern"],
        "alpha_ref": merged["alpha_ref"],
        "playoff_k": merged["playoff_k"],
        "rookie_boost": merged["rookie_boost"],
        "rookie_tau": merged["rookie_tau"],
        "rookie_start": merged["rookie_start"],
        "on_court_mode": merged["on_court_mode"],
        **metric,
        "last_season": int(last_scale["season"]),
        "last_sd": round(float(last_scale["sd"]), 1),
        "last_minutes_wmean": round(float(last_scale["minutes_wmean"]), 1),
        "last_top_rating": round(float(last_scale["top_rating"]), 1),
        "window": cfg["window"],
        "min_games": cfg["min_games"],
        "boundary_start": cfg["boundary_start"],
        "boundary_end": cfg["boundary_end"],
        "engine_seconds": round(elapsed, 1),
    }


def load_checkpoint_state(path):
    """Load a saved end-of-season state into engine initial_state."""
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    return {
        int(row["player_id"]): (float(row["rating"]), int(row["games_played"]),
                                 int(row["games_played_track"]))
        for _, row in df.iterrows()
    }


_WORKER_BOX = None
_WORKER_STATE = None
_WORKER_NAMES = None


def _init_worker(start_season, end_season, checkpoint=None):
    global _WORKER_BOX, _WORKER_STATE, _WORKER_NAMES
    _WORKER_BOX = load_usable_boxscores(start_season, end_season)
    _WORKER_STATE = load_checkpoint_state(checkpoint) if checkpoint else None
    _WORKER_NAMES = load_player_names()



def _coerce_param(v):
    """Keep string grid values (on_court_mode) as-is; numeric to float."""
    return v if isinstance(v, str) else float(v)

def _evaluate_one(task):
    params, cfg = task
    merged = {**DEFAULT_PARAMS, **{k: _coerce_param(v) for k, v in params.items()}}
    merged.setdefault("alpha_ref", merged["alpha_modern"])
    t1 = time.time()
    updates = run_engine(_WORKER_BOX, merged, initial_state=_WORKER_STATE)
    elapsed = time.time() - t1
    if cfg.get("save_snapshots"):
        save_snapshots(updates, merged, cfg, names=_WORKER_NAMES)
    return [
        build_row(updates, merged, _WORKER_BOX, cfg, elapsed, sampling)
        for sampling in cfg["samplings"]
    ]


def expand_grid(spec):
    """Yield parameter dicts from a dict of lists or an explicit list."""
    if isinstance(spec, dict):
        keys = list(spec)
        for combo in itertools.product(*[spec[k] for k in keys]):
            yield dict(zip(keys, combo))
    else:
        for item in spec:
            yield item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=1976)
    parser.add_argument("--end-season", type=int, default=2022)
    parser.add_argument("--boundary-start", type=int, default=1997)
    parser.add_argument("--boundary-end", type=int, default=2021)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min-games", type=int, default=3)
    parser.add_argument("--k", type=float, default=DEFAULT_PARAMS["k"])
    parser.add_argument("--scale", type=float, default=DEFAULT_PARAMS["scale"])
    parser.add_argument("--theta", type=float, default=DEFAULT_PARAMS["theta"])
    parser.add_argument("--h", type=float, default=DEFAULT_PARAMS["h"])
    parser.add_argument("--alpha-modern", type=float,
                        default=DEFAULT_PARAMS["alpha_modern"])
    parser.add_argument("--playoff-k", type=float,
                        default=DEFAULT_PARAMS["playoff_k"])
    parser.add_argument("--rookie-boost", type=float,
                        default=DEFAULT_PARAMS["rookie_boost"])
    parser.add_argument("--rookie-tau", type=float,
                        default=DEFAULT_PARAMS["rookie_tau"])
    parser.add_argument("--rookie-start", type=float,
                        default=DEFAULT_PARAMS["rookie_start"])
    parser.add_argument("--alpha-ref", type=float, default=None,
                        help="CV label blend (default: equals alpha_modern; E-025)")
    parser.add_argument("--on-court-mode", type=str,
                        default=DEFAULT_PARAMS["on_court_mode"],
                        choices=("raw", "team_relative", "on_off"),
                        help="on-court signal variant (R6 A/B)")
    parser.add_argument("--grid", type=str, default=None,
                        help='JSON: list of dicts or dict of lists')
    parser.add_argument("--limit", type=int, default=None,
                        help="only evaluate the first N parameter sets")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--from-parquet", type=str, default=None,
                        help="recompute metric from an existing updates parquet instead of running the engine")
    parser.add_argument("--parallel", type=int, default=1,
                        help="number of worker processes for grid search; each worker loads boxscores independently (default 1)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="end-of-season state parquet to resume from (R4)")
    parser.add_argument("--save-snapshots", action="store_true",
                        help="write season-end/All-Star/top-player CSVs per run (E-025)")
    parser.add_argument("--sampling", type=str, default="season_end",
                        help="rating snapshot point(s), comma-separated: season_end,game20,game41,game60 (R7)")
    parser.add_argument("--out", type=str, default="cv_results.csv",
                        help="output CSV under results/ (default cv_results.csv)")
    args = parser.parse_args()
    if args.checkpoint:
        m = re.search(r"elo_checkpoint_(\d{4})", str(args.checkpoint))
        if m and args.start_season <= int(m.group(1)):
            print(f"checkpoint ends {m.group(1)}; advancing start_season "
                  f"{args.start_season} -> {int(m.group(1)) + 1} to avoid "
                  "replaying pre-checkpoint history")
            args.start_season = int(m.group(1)) + 1

    single = {k: getattr(args, k) for k in DEFAULT_PARAMS}
    if args.grid:
        grid_path = Path(args.grid)
        if grid_path.exists():
            spec = json.loads(grid_path.read_text(encoding="utf-8"))
        else:
            spec = json.loads(args.grid)
        param_sets = list(expand_grid(spec))
    else:
        param_sets = [single]
    if args.alpha_ref is not None:
        single["alpha_ref"] = args.alpha_ref
        for p in param_sets:
            p["alpha_ref"] = args.alpha_ref
    if args.limit:
        param_sets = param_sets[: args.limit]
    if args.from_parquet and len(param_sets) != 1:
        raise ValueError("--from-parquet only reproduces the stored baseline; pass a single parameter set")
    if args.from_parquet and args.parallel != 1:
        raise ValueError("--from-parquet is single-set and does not need --parallel")

    cfg = {
        "window": args.window,
        "min_games": args.min_games,
        "boundary_start": args.boundary_start,
        "boundary_end": args.boundary_end,
        "verbose": args.parallel == 1,
        "save_snapshots": args.save_snapshots,
    }
    samplings = [s.strip() for s in args.sampling.split(",") if s.strip()]
    for s in samplings:
        if s != "season_end" and not re.fullmatch(r"game\d+", s):
            raise ValueError(f"unknown sampling '{s}' (use season_end or game<N>)")
    cfg["samplings"] = samplings

    t0 = time.time()
    stored_updates = None
    box = None
    initial_state = load_checkpoint_state(args.checkpoint) if args.checkpoint else None
    if args.from_parquet:
        stored_updates = pd.read_parquet(args.from_parquet)
        print(f"loaded {len(stored_updates)} updates from {args.from_parquet}")
        if args.end_season < int(stored_updates["season"].max()):
            args.end_season = int(stored_updates["season"].max())
            print(f"extended end_season to {args.end_season} to match parquet")
        box = load_usable_boxscores(args.start_season, args.end_season)
        print(f"loaded {len(box)} rows, {box['game_id'].nunique()} games "
              f"({time.time() - t0:.1f}s)")
    elif args.parallel == 1:
        print(f"loading usable boxscores {args.start_season}-{args.end_season}")
        box = load_usable_boxscores(args.start_season, args.end_season)
        print(f"loaded {len(box)} rows, {box['game_id'].nunique()} games "
              f"({time.time() - t0:.1f}s)")
    else:
        print(f"parallel mode: {args.parallel} workers will load boxscores "
              f"{args.start_season}-{args.end_season} on startup")
        if args.checkpoint:
            print(f"checkpoint: resuming from {args.checkpoint}")

    out_path = RESULTS / args.out
    results = []

    if args.parallel > 1:
        tasks = [(params, cfg) for params in param_sets]
        print(f"grid: {len(tasks)} parameter sets, {args.parallel} workers")
        with ProcessPoolExecutor(
            max_workers=args.parallel,
            initializer=_init_worker,
            initargs=(args.start_season, args.end_season, args.checkpoint),
        ) as pool:
            futures = {
                pool.submit(_evaluate_one, task): task[0]
                for task in tasks
            }
            for idx, fut in enumerate(as_completed(futures), 1):
                params = futures[fut]
                rows = fut.result()
                results.extend(rows)
                RESULTS.mkdir(exist_ok=True)
                pd.DataFrame(results).to_csv(RESULTS / "cv_partial.csv", index=False)
                print(f"[{idx}/{len(tasks)}] done {params}")
                for row in rows:
                    print(f"  [{row['sampling']}] mse={row['mse']:.5f} r={row['r']:.4f} "
                          f"disc={row['discrimination']:.3f} "
                          f"pred_sd={row['pred_sd']:.4f} target_sd={row['target_sd']:.4f} "
                          f"n_players={row['n_players']} ({row['engine_seconds']:.1f}s)")
    else:
        for idx, params in enumerate(param_sets, 1):
            merged = {**DEFAULT_PARAMS, **{k: _coerce_param(v) for k, v in params.items()}}
            merged.setdefault("alpha_ref", merged["alpha_modern"])
            print(f"[{idx}/{len(param_sets)}] {merged}")
            if stored_updates is not None:
                updates = stored_updates
                elapsed = 0.0
            else:
                t1 = time.time()
                updates = run_engine(box, merged, initial_state=initial_state)
                elapsed = time.time() - t1
            print(f"  engine done in {elapsed:.1f}s", flush=True)
            if args.save_snapshots:
                save_snapshots(updates, merged, cfg, names=load_player_names())
            rows = [
                build_row(updates, merged, box, cfg, elapsed, sampling)
                for sampling in cfg["samplings"]
            ]
            results.extend(rows)
            RESULTS.mkdir(exist_ok=True)
            pd.DataFrame(results).to_csv(RESULTS / "cv_partial.csv", index=False)
            for row in rows:
                print(f"  [{row['sampling']}] mse={row['mse']:.5f} r={row['r']:.4f} "
                      f"disc={row['discrimination']:.3f} "
                      f"pred_sd={row['pred_sd']:.4f} target_sd={row['target_sd']:.4f} "
                      f"n_players={row['n_players']} ({elapsed:.1f}s)")

    out_df = pd.DataFrame(results)
    RESULTS.mkdir(exist_ok=True)
    if args.append and out_path.exists():
        existing = pd.read_csv(out_path)
        out_df = pd.concat([existing, out_df], ignore_index=True)
    out_df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out_df)} rows)")
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
