from pathlib import Path

import numpy as np
import pandas as pd

from build_career_curve_clusters import GRID, kmeans, silhouette


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "4.3_career_curve_clusters"
UPDATES = ROOT / "results" / "final" / "final_elo_updates_canonical.csv.gz"
DIAGNOSTICS = ROOT / "results" / "final" / "final_elo_diagnostics_canonical.csv"
PLAYERS = ROOT / "processed_data" / "players.csv"
BASELINE = OUT / "career_curve_membership_boundary_rules.csv"


def make_curve(values: np.ndarray) -> np.ndarray | None:
    smooth = pd.Series(values).rolling(21, center=True, min_periods=1).mean().to_numpy()
    sampled = np.interp(GRID, np.linspace(0.0, 1.0, len(smooth)), smooth)
    if sampled.std() < 1e-9:
        return None
    return (sampled - sampled.mean()) / sampled.std()


def main() -> None:
    players = pd.read_csv(
        PLAYERS, usecols=["personId", "birthDate", "fromYear"]
    ).rename(columns={"personId": "player_id"})
    players["birthDate"] = pd.to_datetime(players["birthDate"], errors="coerce")
    means = pd.read_csv(DIAGNOSTICS).set_index("season")["minutes_wmean"].to_dict()
    updates = pd.read_csv(
        UPDATES,
        usecols=["player_id", "game_date", "season", "rating_after"],
        parse_dates=["game_date"],
    ).sort_values(["player_id", "game_date"])
    updates["relative_elo"] = updates["rating_after"] - updates["season"].map(means)
    latest = int(updates["season"].max())
    meta = updates.groupby("player_id").agg(
        games=("rating_after", "size"),
        seasons=("season", "nunique"),
        first_season=("season", "min"),
        last_season=("season", "max"),
        last_game=("game_date", "max"),
    ).reset_index().merge(players, on="player_id", how="left").set_index("player_id")
    meta["age_at_last"] = (meta["last_game"] - meta["birthDate"]).dt.days / 365.2425
    meta["full_start"] = (meta["first_season"] >= 1976) & (meta["first_season"] == meta["fromYear"])
    meta["eligible_end"] = (meta["last_season"] < latest) | (
        (meta["last_season"] == latest) & (meta["age_at_last"] >= 33.0)
    )
    candidate_ids = meta[meta["full_start"] & meta["eligible_end"]].index

    curves = {}
    for player_id, group in updates[updates["player_id"].isin(candidate_ids)].groupby("player_id"):
        value = make_curve(group["relative_elo"].to_numpy())
        if value is not None:
            curves[int(player_id)] = value

    season_games = updates.groupby(["player_id", "season"]).size().rename("games").reset_index()
    baseline = pd.read_csv(BASELINE).set_index("player_id")["cluster"].astype(int)
    rows = []
    for season_game_min in [5, 10, 15, 20, 25, 30, 40, 50]:
        counts = (
            season_games[season_games["games"] >= season_game_min]
            .groupby("player_id").size()
        )
        for season_count_min in [3, 4, 5, 6, 7, 8, 9, 10]:
            ids = sorted(
                int(i) for i in candidate_ids
                if counts.get(i, 0) >= season_count_min and int(i) in curves
            )
            x = np.vstack([curves[i] for i in ids])
            labels, centers, inertia = kmeans(x, 3, seed=42, n_init=60)
            order = np.argsort([GRID[int(np.argmax(c))] for c in centers])
            centers = centers[order]
            remap = {int(old): int(new) + 1 for new, old in enumerate(order)}
            labels = np.asarray([remap[int(v)] for v in labels])
            assignment = dict(zip(ids, labels))
            shared = sorted(set(ids) & set(baseline.index.astype(int)))
            rows.append(
                {
                    "games_per_qualifying_season": season_game_min,
                    "minimum_qualifying_seasons": season_count_min,
                    "players": len(ids),
                    "silhouette": silhouette(x, labels),
                    "baseline_shared": len(shared),
                    "baseline_stability": np.mean(
                        [assignment[i] == int(baseline.loc[i]) for i in shared]
                    ),
                    "cluster1_peak": GRID[np.argmax(centers[0])],
                    "cluster2_peak": GRID[np.argmax(centers[1])],
                    "cluster3_peak": GRID[np.argmax(centers[2])],
                    "cluster1_players": int((labels == 1).sum()),
                    "cluster2_players": int((labels == 2).sum()),
                    "cluster3_players": int((labels == 3).sum()),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "meaningful_season_grid_with_boundary_rules.csv", index=False)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
