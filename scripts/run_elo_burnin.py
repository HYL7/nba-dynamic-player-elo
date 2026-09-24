"""Run a fixed-parameter Elo window over clean_data for burn-in observation.

Usage:
    python scripts/run_elo_burnin.py [start_season] [end_season] [init_mode]

init_mode: "uniform" (everyone 1500, everyone gets rookie K-boost, legacy
smoke-test mode) or "draft" (draft-pick initial rating; every debutant gets
the rookie K-boost per D-022). Default is "draft".
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.elo.engine import EloEngine, build_draft_rating_map

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


def build_draft_inputs():
    """Return (draft_ratings, rookie_ids) from players_clean + draft history.

    rookie_ids is always None per D-022: every player who debuts inside the
    Elo data window gets the rookie K-boost, because the window starts at
    1976-77 and the system has no prior for anyone before that season.
    draftYear is not used for boost eligibility (ABA veterans such as Moses
    Malone have no NBA draft record); it still feeds the draft-pick initial
    rating when available.
    """
    players = pd.read_csv("clean_data/players_clean.csv", low_memory=False)
    draft = pd.read_csv("processed_data/draft_history.csv", usecols=["person_id", "overall_pick"], low_memory=False)
    draft = draft.dropna(subset=["overall_pick"])
    picks = draft.groupby("person_id")["overall_pick"].min()
    players = players.merge(picks.rename("overall_pick"), left_on="personId", right_index=True, how="left")

    rating_map = {}
    for row in players.itertuples(index=False):
        pid = int(row.personId)
        if pd.isna(row.overall_pick):
            rating_map[pid] = 1350.0
            continue
        rating = 1580.0 - (float(row.overall_pick) - 1.0) * 5.0
        rating_map[pid] = max(rating, 1350.0)

    return rating_map, None


def main():
    start_season = int(sys.argv[1]) if len(sys.argv) > 1 else 1976
    end_season = int(sys.argv[2]) if len(sys.argv) > 2 else start_season + 1
    init_mode = sys.argv[3] if len(sys.argv) > 3 else "draft"

    t0 = time.time()
    print(f"loading usable boxscores for {start_season}-{end_season}")
    box = load_usable_boxscores(start_season, end_season)
    print(f"loaded {len(box)} rows, {box['game_id'].nunique()} games "
          f"({time.time() - t0:.1f}s)")

    draft_ratings, rookie_ids = build_draft_inputs()
    if init_mode == "draft":
        print(f"draft mode: {len(draft_ratings)} ratings, "
              f"all debutants get rookie K-boost (D-022)")
    else:
        draft_ratings, rookie_ids = None, None

    engine = EloEngine(
        k=20.0, theta=0.3, home_advantage=70.0, alpha=1.0,
        rookie_boost=3.0, rookie_tau=25.0,
        init_mode=init_mode, draft_ratings=draft_ratings,
        rookie_ids=rookie_ids,
    )
    t1 = time.time()
    updates, states, inits = engine.run(box)
    print(f"engine ran in {time.time() - t1:.1f}s: "
          f"{len(updates)} updates, {len(inits)} inits")

    per_game = updates.groupby("game_id")["delta_adj"].sum()
    print(f"max |sum(delta_adj)| per game: {per_game.abs().max():.3e}")
    print(f"NaN in key columns: {int(updates[['rating_before', 'rating_after', 'perf_i', 'minutes_share']].isna().sum().sum())}")

    final = updates.sort_values(["game_date", "game_id"]).groupby("player_id").tail(1)
    final = final.sort_values("rating_after", ascending=False)
    print("\ntop 12 final ratings:")
    for row in final.head(12).itertuples(index=False):
        print(f"  {int(row.player_id):>8}  {row.rating_after:8.1f}")

    suffix = f"{start_season}-{end_season}"
    RESULTS.mkdir(exist_ok=True)
    updates.to_csv(RESULTS / f"burnin_updates_{suffix}_{init_mode}.csv", index=False)
    states.to_csv(RESULTS / f"burnin_states_{suffix}_{init_mode}.csv", index=False)
    inits.to_csv(RESULTS / f"burnin_inits_{suffix}_{init_mode}.csv", index=False)
    print(f"wrote results/ ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
