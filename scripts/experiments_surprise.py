"""15-year surprise-version sweep for the Dynamic Player Elo.

Runs 1976-77 through 1990-91 with draft init mode (D-022) and reports, per
season: simple mean, minute-weighted mean, SD, mean of players with >= 1000
minutes, and top rating. Also reports selected star season-end ratings and
per-game zero-sum check.

Usage:
    python scripts/experiments_surprise.py
"""

import time
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.elo.engine import EloEngine

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

WATCH = {
    76003: "Kareem Abdul-Jabbar",
    77449: "Moses Malone",
    76681: "Julius Erving",
    76804: "George Gervin",
}


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


def build_draft_inputs():
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
            rating_map[pid] = 1350.0
            continue
        rating = 1580.0 - (float(row.overall_pick) - 1.0) * 5.0
        rating_map[pid] = max(rating, 1350.0)
    return rating_map


def season_metrics(updates):
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


def run_one(box, draft_ratings, k, scale, anchor, label):
    t0 = time.time()
    engine = EloEngine(
        k=k, theta=0.3, home_advantage=70.0, alpha=1.0,
        rookie_boost=3.0, rookie_tau=25.0,
        init_mode="draft", draft_ratings=draft_ratings,
        rookie_ids=None, record_states=False,
        surprise_scale=scale, surprise_anchor=anchor,
    )
    updates, _, _ = engine.run(box)
    elapsed = time.time() - t0
    updates = updates.merge(
        box[["game_id", "personId", "scaled_minutes"]].rename(
            columns={"personId": "player_id"}
        ),
        on=["game_id", "player_id"],
        how="left",
    )

    per_game = updates.groupby("game_id")["delta_adj"].sum()
    metrics = season_metrics(updates)
    watch = []
    for pid, name in WATCH.items():
        row = updates[updates["player_id"] == pid].sort_values(
            ["game_date", "game_id"]
        ).tail(1)
        if not row.empty:
            watch.append(f"{name}:{row['rating_after'].iloc[0]:.0f}")

    print(f"\n=== {label} (k={k}, scale={scale}, anchor={anchor}) [{elapsed:.0f}s] ===")
    print("season,n,mean,wmean,sd,ge1000,top")
    for row in metrics.itertuples(index=False):
        print(
            f"{row.season},{row.n_players},{row.mean:.1f},{row.minutes_wmean:.1f},"
            f"{row.sd:.1f},{row.mean_ge1000min:.1f},{row.top_rating:.1f}"
        )
    print(f"max |sum(delta_adj)| per game: {per_game.abs().max():.3e}")
    print("stars end of window: " + "; ".join(watch))
    return metrics


def main():
    t0 = time.time()
    start_season = int(sys.argv[4]) if len(sys.argv) > 4 else 1976
    end_season = int(sys.argv[5]) if len(sys.argv) > 5 else 1990
    print(f"loading usable boxscores {start_season}-{end_season}")
    box = load_usable_boxscores(start_season, end_season)
    print(f"loaded {len(box)} rows, {box['game_id'].nunique()} games ({time.time() - t0:.0f}s)")

    draft_ratings = build_draft_inputs()
    print(f"draft ratings: {len(draft_ratings)}")

    results = {}
    combos = [
        (10, 400.0, "fixed"),
        (10, 400.0, "game"),
        (10, 400.0, "league"),
        (15, 400.0, "fixed"),
        (10, 300.0, "fixed"),
        (10, 500.0, "fixed"),
        (10, 1500.0, "fixed"),
    ]
    if len(sys.argv) > 1:
        combos = [(
            float(sys.argv[1]),
            float(sys.argv[2]),
            sys.argv[3],
        )]
    for k, scale, anchor in combos:
        label = f"surprise_{anchor}_k{k}_s{int(scale)}"
        results[label] = run_one(box, draft_ratings, k, scale, anchor, label)


if __name__ == "__main__":
    main()
