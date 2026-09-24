from pathlib import Path

import numpy as np
import pandas as pd

from build_career_curve_clusters import GRID, kmeans, silhouette


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "4.3_career_curve_clusters"
UPDATES = ROOT / "results" / "final" / "final_elo_updates_canonical.csv.gz"
DIAGNOSTICS = ROOT / "results" / "final" / "final_elo_diagnostics_canonical.csv"
PLAYERS = ROOT / "processed_data" / "players.csv"


def make_curve(values: np.ndarray) -> np.ndarray | None:
    smooth = pd.Series(values).rolling(21, center=True, min_periods=1).mean().to_numpy()
    sampled = np.interp(GRID, np.linspace(0.0, 1.0, len(smooth)), smooth)
    if sampled.std() < 1e-9:
        return None
    return (sampled - sampled.mean()) / sampled.std()


def main() -> None:
    players = pd.read_csv(
        PLAYERS,
        usecols=["personId", "firstName", "lastName", "fromYear", "toYear"],
    ).rename(columns={"personId": "player_id"})
    players["full_name"] = players["firstName"].fillna("") + " " + players["lastName"].fillna("")
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
    keep = meta[
        (meta["games"] >= 300)
        & (meta["seasons"] >= 5)
        & (meta["first_season"] >= 1976)
        & (meta["first_season"] == meta["fromYear"])
        & (meta["last_season"] <= 2023)
        & (meta["last_season"] >= meta["toYear"] - 1)
    ].index

    ids, curves = [], []
    for player_id, group in updates[updates["player_id"].isin(keep)].groupby("player_id"):
        value = make_curve(group["relative_elo"].to_numpy())
        if value is not None:
            ids.append(int(player_id))
            curves.append(value)
    x = np.vstack(curves)

    selection_rows = []
    for k in range(2, 9):
        labels, centers, inertia = kmeans(x, k, seed=42, n_init=60)
        score = silhouette(x, labels)
        order = np.argsort([GRID[int(np.argmax(c))] for c in centers])
        centers = centers[order]
        remap = {int(old): int(new) + 1 for new, old in enumerate(order)}
        ordered_labels = np.asarray([remap[int(v)] for v in labels])
        row = {"k": k, "silhouette": score, "inertia": inertia}
        for cluster in range(1, k + 1):
            center = centers[cluster - 1]
            row[f"c{cluster}_n"] = int((ordered_labels == cluster).sum())
            row[f"c{cluster}_peak"] = float(GRID[np.argmax(center)])
            row[f"c{cluster}_start"] = float(center[0])
            row[f"c{cluster}_end"] = float(center[-1])
        selection_rows.append(row)

        center_rows = []
        for cluster, center in enumerate(centers, start=1):
            center_row = {
                "k": k,
                "cluster": cluster,
                "players": int((ordered_labels == cluster).sum()),
                "peak_progress": float(GRID[np.argmax(center)]),
            }
            center_row.update({f"p{int(p * 100):03d}": float(v) for p, v in zip(GRID, center)})
            center_rows.append(center_row)
        pd.DataFrame(center_rows).to_csv(OUT / f"career_curve_centers_300g_5s_k{k}.csv", index=False)

        membership = pd.DataFrame({"player_id": ids, "cluster": ordered_labels}).merge(
            players[["player_id", "full_name"]], on="player_id", how="left"
        )
        membership.to_csv(OUT / f"career_curve_membership_300g_5s_k{k}.csv", index=False)

    selection = pd.DataFrame(selection_rows)
    selection.to_csv(OUT / "career_curve_k_selection_300g_5s.csv", index=False)
    print(selection.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
