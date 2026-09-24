"""Pure Elo math used by the Dynamic Player Elo pipeline.

All functions are deterministic and side-effect free so they can be unit
tested with synthetic data. The sequential game engine lives in engine.py.
"""

import numpy as np
import pandas as pd

GAME_SCORE_WEIGHTS = {
    "points": 1.0,
    "fieldGoalsMade": 0.4,
    "fieldGoalsAttempted": -0.7,
    "freeThrowsAttempted": -0.4,
    "freeThrowsMade": 0.4,
    "reboundsOffensive": 0.7,
    "reboundsDefensive": 0.3,
    "assists": 0.7,
    "steals": 1.0,
    "blocks": 0.7,
    "foulsPersonal": -0.4,
    "turnovers": -1.0,
}

REQUIRED_GAME_SCORE_COLS = [
    "points",
    "fieldGoalsMade",
    "fieldGoalsAttempted",
    "freeThrowsAttempted",
    "freeThrowsMade",
    "reboundsOffensive",
    "reboundsDefensive",
    "assists",
    "steals",
    "blocks",
    "foulsPersonal",
    "turnovers",
]


def game_score(row):
    """Hollinger GameScore from one boxscore row (dict or pandas Series)."""
    return sum(row[col] * weight for col, weight in GAME_SCORE_WEIGHTS.items())


def zscore(values, sigma_floor=1.0):
    """Standardize with a sigma floor to avoid blow-up in low-variance games."""
    values = np.asarray(values, dtype=float)
    mu = values.mean()
    sigma = values.std(ddof=0)
    sigma_eff = max(sigma, sigma_floor)
    return (values - mu) / sigma_eff


def on_court_rate_shrunk(plus_minus, minutes):
    """Per-minute plus/minus shrunk toward 0 for low-minute players."""
    raw = plus_minus / minutes
    return raw * (minutes / (minutes + 5.0))


def _team_on_court_baseline(frame):
    """Per-team game totals (margin, minutes) for relative on-court variants."""
    agg = frame.groupby("playerteamId")[["plusMinusPoints", "minutes"]].sum()
    return (
        frame["playerteamId"].map(agg["plusMinusPoints"]),
        frame["playerteamId"].map(agg["minutes"]),
    )


def on_court_rate_variant(frame, mode):
    """Relative on-court rates for a full game frame (R6 on-off A/B).

    mode:
      team_relative : on_rate - team game-average net rate
      on_off        : on_rate - off_rate (team margin while the player sits)
    Both keep the existing m/(m+5) shrinkage; on_off additionally shrinks by
    off_minutes/(off_minutes+5) and returns 0 when the player plays the whole
    game (no off-court sample)."""
    margin_map, team_min_map = _team_on_court_baseline(frame)
    pm = frame["plusMinusPoints"].to_numpy(dtype=float)
    mins = frame["minutes"].to_numpy(dtype=float)
    margin = margin_map.to_numpy(dtype=float)
    team_min = team_min_map.to_numpy(dtype=float)
    on_rate = np.where(mins > 0, pm / np.maximum(mins, 1e-9), 0.0)
    if mode == "team_relative":
        rel = on_rate - margin / np.maximum(team_min, 1e-9)
        return rel * (mins / (mins + 5.0))
    if mode == "on_off":
        off_min = np.maximum(team_min - mins, 0.0)
        off_rate = np.where(
            off_min > 0,
            (margin - pm) / np.maximum(off_min, 1e-9),
            0.0,
        )
        diff = on_rate - off_rate
        shrink = (mins / (mins + 5.0)) * (off_min / (off_min + 5.0))
        return diff * shrink
    raise ValueError(f"unknown on_court_mode: {mode}")


def perf_i(game_score_values, on_court_values, alpha, sigma_floor=1.0):
    """Blend z-scored GameScore and z-scored on-court signal.

    on_court_values may be None/NaN when plus/minus is unavailable; the
    on-court term then contributes 0 (equivalent to alpha=1 for that game).
    """
    gs = zscore(game_score_values, sigma_floor)
    if on_court_values is None:
        return gs
    oc = np.asarray(on_court_values, dtype=float)
    if np.isnan(oc).all():
        return gs
    oc = np.where(np.isnan(oc), 0.0, oc)
    oc_z = zscore(oc, sigma_floor)
    return alpha * gs + (1.0 - alpha) * oc_z


def surprise_perf(perf, rating, anchor, scale):
    # Perf relative to the level implied by the player's rating.
    return perf - (rating - anchor) / scale


def expected_win_probability(rating_diff, home_advantage=0.0):
    """Expected win probability of team A against team B.

    rating_diff = R_A - R_B. A positive home_advantage favors team A.
    """
    exponent = -(rating_diff + home_advantage) / 400.0
    return 1.0 / (1.0 + 10.0**exponent)


def k_effective(k_base, games_played, rookie_boost=1.0, tau=20.0):
    """Rookie K multiplier that decays to 1.0 as games accumulate."""
    decay = np.exp(-games_played / tau) if tau > 0 else 0.0
    return k_base * (1.0 + (rookie_boost - 1.0) * decay)


def elo_delta(k, minutes_share, perf, s_team, e_team, theta):
    """Raw rating change before the per-game zero-sum adjustment."""
    return k * minutes_share * (perf + theta * (s_team - e_team))


def apply_zero_sum(deltas):
    """Subtract the game mean so adjusted deltas sum to exactly zero."""
    deltas = np.asarray(deltas, dtype=float)
    return deltas - deltas.mean()


def team_rating_average(player_ratings, minutes_shares):
    """Minutes-weighted average rating for one team."""
    ratings = np.asarray(player_ratings, dtype=float)
    shares = np.asarray(minutes_shares, dtype=float)
    return float(np.dot(ratings, shares) / shares.sum())


def prepare_game_score_frame(frame):
    """Return a DataFrame with GameScore, on-court rate and perf_i computed.

    frame must contain the boxscore columns plus a plusMinusPoints column.
    """
    frame = frame.copy()
    missing = [c for c in REQUIRED_GAME_SCORE_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing GameScore columns: {missing}")
    for col in REQUIRED_GAME_SCORE_COLS:
        frame[col] = frame[col].fillna(0.0).astype(float)
    frame["game_score"] = frame.apply(game_score, axis=1)
    frame["on_court_rate"] = np.where(
        frame["plusMinusPoints"].isna() | (frame["minutes"] <= 0),
        np.nan,
        on_court_rate_shrunk(frame["plusMinusPoints"], frame["minutes"]),
    )
    return frame


def _sanity_check_game_frame(frame, alpha):
    if frame["minutes_share"].sum() <= 0:
        raise ValueError("game has no positive minutes_share")
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in [0, 1]")


def compute_game_performance(frame, alpha, sigma_floor=1.0, on_court_mode="raw"):
    """Compute per-player z-scores and blended perf for one game.

    Returns a copy of frame with columns:
    game_score, on_court_rate, z_game_score, z_on_court_rate, perf_i.
    z_on_court_rate is NaN when on-court signal is unavailable.
    """
    _sanity_check_game_frame(frame, alpha)
    out = prepare_game_score_frame(frame)
    if on_court_mode != "raw":
        mask = frame["plusMinusPoints"].notna() & (frame["minutes"] > 0)
        out["on_court_rate"] = np.where(mask, on_court_rate_variant(frame, on_court_mode), np.nan)
    oc_available = out["on_court_rate"].notna()
    on_court_values = out["on_court_rate"].where(oc_available, np.nan)
    out["z_game_score"] = zscore(out["game_score"], sigma_floor)
    out["z_on_court_rate"] = zscore(
        np.where(oc_available, on_court_values, 0.0), sigma_floor
    ) * oc_available.astype(float)
    out["perf_i"] = perf_i(
        out["game_score"], on_court_values, alpha, sigma_floor
    )
    return out
