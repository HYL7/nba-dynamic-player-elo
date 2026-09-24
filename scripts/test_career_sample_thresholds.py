from pathlib import Path

import numpy as np
import pandas as pd

from build_career_curve_clusters import GRID, kmeans, silhouette


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "4.3_career_curve_clusters"
UPDATES = ROOT / "results" / "final" / "final_elo_updates_canonical.csv.gz"
DIAGNOSTICS = ROOT / "results" / "final" / "final_elo_diagnostics_canonical.csv"
PLAYERS = ROOT / "processed_data" / "players.csv"
BASE = OUT / "career_curve_cluster_membership.csv"


def make_curve(values: np.ndarray) -> np.ndarray | None:
    smooth = pd.Series(values).rolling(21, center=True, min_periods=1).mean().to_numpy()
    sampled = np.interp(GRID, np.linspace(0.0, 1.0, len(smooth)), smooth)
    if sampled.std() < 1e-9:
        return None
    return (sampled - sampled.mean()) / sampled.std()


def fit_ordered(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    labels, centers, _ = kmeans(x, 3)
    order = np.argsort([GRID[int(np.argmax(c))] for c in centers])
    remap = {int(old): int(new) + 1 for new, old in enumerate(order)}
    return np.asarray([remap[int(v)] for v in labels]), centers[order], silhouette(x, labels)


def main() -> None:
    base = pd.read_csv(BASE).set_index("player_id")
    players = pd.read_csv(
        PLAYERS, usecols=["personId", "fromYear", "toYear"]
    ).rename(columns={"personId": "player_id"})
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
    ).reset_index().merge(players, on="player_id", how="left").set_index("player_id")
    completed = meta[
        (meta["first_season"] >= 1976)
        & (meta["first_season"] == meta["fromYear"])
        & (meta["last_season"] <= 2023)
        & (meta["last_season"] >= meta["toYear"] - 1)
    ]
    curves = {}
    for player_id, group in updates[updates["player_id"].isin(completed.index)].groupby("player_id"):
        value = make_curve(group["relative_elo"].to_numpy())
        if value is not None:
            curves[int(player_id)] = value

    rows = []
    for min_games in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
        for min_seasons in [5, 6, 7, 8, 9, 10]:
            ids = [
                int(i) for i, r in completed.iterrows()
                if r["games"] >= min_games and r["seasons"] >= min_seasons and int(i) in curves
            ]
            x = np.vstack([curves[i] for i in ids])
            labels, centers, sil = fit_ordered(x)
            assigned = dict(zip(ids, labels))
            if min_games == 0 and min_seasons == 5:
                five_season_membership = pd.DataFrame(
                    {"player_id": ids, "cluster": labels}
                ).merge(
                    completed[["games", "seasons"]].reset_index(),
                    on="player_id",
                    how="left",
                )
                five_season_membership.to_csv(
                    OUT / "career_curve_cluster_membership_min_5_seasons.csv", index=False
                )
            shared = sorted(set(ids) & set(base.index.astype(int)))
            stability = np.mean([assigned[i] == int(base.loc[i, "cluster"]) for i in shared])
            rows.append(
                {
                    "min_games": min_games,
                    "min_seasons": min_seasons,
                    "players": len(ids),
                    "silhouette": sil,
                    "baseline_shared_players": len(shared),
                    "baseline_label_stability": stability,
                    "cluster1_peak": GRID[np.argmax(centers[0])],
                    "cluster2_peak": GRID[np.argmax(centers[1])],
                    "cluster3_peak": GRID[np.argmax(centers[2])],
                    "cluster1_players": int((labels == 1).sum()),
                    "cluster2_players": int((labels == 2).sum()),
                    "cluster3_players": int((labels == 3).sum()),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "career_sample_threshold_sensitivity.csv", index=False)

    # Alternative tenure definition: count only seasons with meaningful participation.
    season_games = updates.groupby(["player_id", "season"]).size().rename("season_games").reset_index()
    meaningful_rows = []
    for games_in_season in [5, 10, 15, 20, 25, 30, 40, 50]:
        qualifying = (
            season_games[season_games["season_games"] >= games_in_season]
            .groupby("player_id").size()
        )
        ids = [
            int(i) for i in completed.index
            if qualifying.get(i, 0) >= 5 and int(i) in curves
        ]
        x = np.vstack([curves[i] for i in ids])
        labels, centers, sil = fit_ordered(x)
        assigned = dict(zip(ids, labels))
        shared = sorted(set(ids) & set(base.index.astype(int)))
        meaningful_rows.append(
            {
                "minimum_games_in_qualifying_season": games_in_season,
                "required_qualifying_seasons": 5,
                "players": len(ids),
                "silhouette": sil,
                "baseline_shared_players": len(shared),
                "baseline_label_stability": np.mean(
                    [assigned[i] == int(base.loc[i, "cluster"]) for i in shared]
                ),
                "cluster1_peak": GRID[np.argmax(centers[0])],
                "cluster2_peak": GRID[np.argmax(centers[1])],
                "cluster3_peak": GRID[np.argmax(centers[2])],
                "cluster1_players": int((labels == 1).sum()),
                "cluster2_players": int((labels == 2).sum()),
                "cluster3_players": int((labels == 3).sum()),
            }
        )
    meaningful_df = pd.DataFrame(meaningful_rows)
    meaningful_df.to_csv(OUT / "meaningful_season_threshold_sensitivity.csv", index=False)

    corr = completed[["games", "seasons"]].corr().iloc[0, 1]
    conditions = {
        "both": ((completed["games"] >= 600) & (completed["seasons"] >= 8)).sum(),
        "games_only": ((completed["games"] >= 600) & (completed["seasons"] < 8)).sum(),
        "seasons_only": ((completed["games"] < 600) & (completed["seasons"] >= 8)).sum(),
        "neither": ((completed["games"] < 600) & (completed["seasons"] < 8)).sum(),
    }
    game_survival = pd.DataFrame(
        [
            {
                "min_games": threshold,
                "players": int((completed["games"] >= threshold).sum()),
                "share": float((completed["games"] >= threshold).mean()),
            }
            for threshold in range(0, 1301, 50)
        ]
    )
    game_survival["lost_next_50"] = game_survival["players"] - game_survival["players"].shift(-1)
    season_survival = pd.DataFrame(
        [
            {
                "min_seasons": threshold,
                "players": int((completed["seasons"] >= threshold).sum()),
                "share": float((completed["seasons"] >= threshold).mean()),
            }
            for threshold in range(1, 21)
        ]
    )
    season_survival["lost_next_season"] = season_survival["players"] - season_survival["players"].shift(-1)
    game_survival.to_csv(OUT / "completed_career_game_threshold_distribution.csv", index=False)
    season_survival.to_csv(OUT / "completed_career_season_threshold_distribution.csv", index=False)
    print(f"completed_pool={len(completed)} games_seasons_corr={corr:.4f} conditions={conditions}")
    selected = result[
        ((result["min_games"].isin([400, 500, 600, 700, 800])) & (result["min_seasons"] == 8))
        | ((result["min_games"] == 600) & result["min_seasons"].isin([5, 6, 7, 8, 9, 10]))
    ].sort_values(["min_games", "min_seasons"])
    print(selected.round(4).to_string(index=False))
    print("\nGAME THRESHOLD DISTRIBUTION (300-900)")
    print(game_survival[game_survival["min_games"].between(300, 900)].round(3).to_string(index=False))
    print("\nSEASON THRESHOLD DISTRIBUTION (5-15)")
    print(season_survival[season_survival["min_seasons"].between(5, 15)].round(3).to_string(index=False))
    print("\nFIVE MEANINGFUL SEASONS")
    print(meaningful_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
