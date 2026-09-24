from pathlib import Path

import numpy as np
import pandas as pd

from build_career_curve_clusters import GRID, kmeans, silhouette


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "4.3_career_curve_clusters"
UPDATES = ROOT / "results" / "final" / "final_elo_updates_canonical.csv.gz"
DIAGNOSTICS = ROOT / "results" / "final" / "final_elo_diagnostics_canonical.csv"
PLAYERS = ROOT / "processed_data" / "players.csv"
BASE_MEMBERSHIP = OUT / "career_curve_cluster_membership.csv"


def curve(values: np.ndarray) -> np.ndarray | None:
    smooth = pd.Series(values).rolling(21, center=True, min_periods=1).mean().to_numpy()
    sampled = np.interp(GRID, np.linspace(0.0, 1.0, len(smooth)), smooth)
    if sampled.std() < 1e-9:
        return None
    return (sampled - sampled.mean()) / sampled.std()


def main() -> None:
    base = pd.read_csv(BASE_MEMBERSHIP).set_index("player_id")
    players = pd.read_csv(
        PLAYERS,
        usecols=["personId", "birthDate", "fromYear", "toYear"],
    ).rename(columns={"personId": "player_id"})
    players["birthDate"] = pd.to_datetime(players["birthDate"], errors="coerce")
    means = pd.read_csv(DIAGNOSTICS).set_index("season")["minutes_wmean"].to_dict()
    updates = pd.read_csv(
        UPDATES,
        usecols=["player_id", "game_date", "season", "rating_after"],
        parse_dates=["game_date"],
    ).sort_values(["player_id", "game_date"])
    updates["relative_elo"] = updates["rating_after"] - updates["season"].map(means)
    meta = updates.groupby("player_id").agg(
        games=("rating_after", "size"),
        seasons=("season", "nunique"),
        first_season=("season", "min"),
        last_season=("season", "max"),
        last_game=("game_date", "max"),
    ).reset_index().merge(players, on="player_id", how="left").set_index("player_id")
    meta["age_at_last"] = (meta["last_game"] - meta["birthDate"]).dt.days / 365.2425
    eligible = meta[
        (meta["games"] >= 600)
        & (meta["seasons"] >= 8)
        & (meta["first_season"] >= 1976)
        & (meta["first_season"] == meta["fromYear"])
    ]
    active = eligible[eligible["last_season"] >= 2024]
    completed_ids = set(base.index.astype(int))
    candidate_ids = completed_ids | set(active.index.astype(int))
    curves = {}
    for player_id, group in updates[updates["player_id"].isin(candidate_ids)].groupby("player_id"):
        value = curve(group["relative_elo"].to_numpy())
        if value is not None:
            curves[int(player_id)] = value

    rows = []
    for min_age in np.arange(29.0, 40.0, 0.5):
        active_ids = [int(i) for i in active[active["age_at_last"] >= min_age].index if int(i) in curves]
        ids = sorted(completed_ids | set(active_ids))
        x = np.vstack([curves[i] for i in ids])
        labels, centers, inertia = kmeans(x, 3)
        order = np.argsort([GRID[int(np.argmax(center))] for center in centers])
        remap = {int(old): int(new) + 1 for new, old in enumerate(order)}
        assigned = {player_id: remap[int(label)] for player_id, label in zip(ids, labels)}
        stable = np.mean([assigned[i] == int(base.loc[i, "cluster"]) for i in completed_ids])
        rows.append(
            {
                "active_min_age": min_age,
                "active_players": len(active_ids),
                "total_players": len(ids),
                "completed_label_stability": stable,
                "silhouette": silhouette(x, labels),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "active_age_threshold_cluster_stability.csv", index=False)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
