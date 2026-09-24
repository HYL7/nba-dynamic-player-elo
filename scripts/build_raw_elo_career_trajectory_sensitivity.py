from pathlib import Path

import numpy as np
import pandas as pd

from build_career_curve_clusters import GRID, kmeans, silhouette
from build_final_career_trajectory_results import draw_chart


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "analysis" / "4.3_career_curve_clusters" / "final_40_or_8_rule"
MEMBERSHIP = BASE / "player_career_trajectory_classification.csv"
UPDATES = ROOT / "results" / "final" / "final_elo_updates_canonical.csv.gz"
PLAYERS = ROOT / "processed_data" / "players.csv"
SMOOTHING_GAMES = 21
CLUSTER_NAMES = {
    1: "Early peak / gradual decline",
    2: "Conventional mid-career peak",
    3: "Late peak / sustained rise",
}


def prepare_raw_curve(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    group = group.sort_values("game_date").reset_index(drop=True)
    smoothed = group["rating_after"].rolling(
        SMOOTHING_GAMES, center=True, min_periods=1
    ).mean().to_numpy()
    progress = np.linspace(0.0, 1.0, len(group))
    sampled = np.interp(GRID, progress, smoothed)
    standardized = (sampled - sampled.mean()) / sampled.std()
    sampled_age = np.interp(GRID, progress, group["age"].to_numpy())
    return standardized, sampled_age


def main() -> None:
    original = pd.read_csv(MEMBERSHIP)
    eligible_ids = set(original["player_id"])
    players = pd.read_csv(
        PLAYERS, usecols=["personId", "birthDate"]
    ).rename(columns={"personId": "player_id"})
    players["birthDate"] = pd.to_datetime(players["birthDate"], errors="coerce")
    birth_map = players.set_index("player_id")["birthDate"]

    updates = pd.read_csv(
        UPDATES,
        usecols=["player_id", "game_date", "rating_after"],
        parse_dates=["game_date"],
    )
    updates = updates[updates["player_id"].isin(eligible_ids)].copy()
    updates["birthDate"] = updates["player_id"].map(birth_map)
    updates["age"] = (updates["game_date"] - updates["birthDate"]).dt.days / 365.2425

    ids, curves, age_curves = [], [], []
    for player_id, group in updates.dropna(subset=["age"]).groupby("player_id", sort=True):
        curve, ages = prepare_raw_curve(group)
        ids.append(int(player_id))
        curves.append(curve)
        age_curves.append(ages)
    x = np.vstack(curves)
    age_x = np.vstack(age_curves)

    selection_rows = []
    fits = {}
    for k in range(2, 9):
        labels, centers, inertia = kmeans(x, k, seed=42, n_init=80)
        selection_rows.append(
            {"k": k, "silhouette": silhouette(x, labels), "inertia": inertia}
        )
        fits[k] = (labels, centers)
    pd.DataFrame(selection_rows).to_csv(
        BASE / "cluster_count_selection_raw_elo.csv", index=False
    )

    labels, centers = fits[3]
    order = np.argsort([GRID[int(np.argmax(center))] for center in centers])
    centers = centers[order]
    remap = {int(old): int(new) + 1 for new, old in enumerate(order)}
    labels = np.asarray([remap[int(label)] for label in labels])

    raw_membership = pd.DataFrame(
        {
            "player_id": ids,
            "raw_elo_cluster": labels,
            "raw_elo_cluster_name": [CLUSTER_NAMES[int(label)] for label in labels],
            "distance_to_raw_cluster_center": [
                float(np.sqrt(np.square(curve - centers[label - 1]).sum()))
                for curve, label in zip(x, labels)
            ],
        }
    ).merge(
        original[["player_id", "full_name", "cluster", "cluster_name"]],
        on="player_id",
        how="left",
    )
    raw_membership.to_csv(
        BASE / "player_career_trajectory_classification_raw_elo.csv", index=False
    )
    pd.crosstab(
        raw_membership["cluster"], raw_membership["raw_elo_cluster"]
    ).to_csv(BASE / "relative_vs_raw_cluster_crosstab.csv")

    summary_rows = []
    for cluster in range(1, 4):
        mask = labels == cluster
        peak_index = int(np.argmax(centers[cluster - 1]))
        summary_rows.append(
            {
                "cluster": cluster,
                "players": int(mask.sum()),
                "center_peak_progress": float(GRID[peak_index]),
                "age_at_center_peak": float(np.nanmedian(age_x[mask, peak_index])),
                "center_start": float(centers[cluster - 1, 0]),
                "center_end": float(centers[cluster - 1, -1]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(BASE / "cluster_summary_raw_elo.csv", index=False)
    draw_chart(
        curves=x,
        labels=labels,
        centers=centers,
        age_grid=np.nanmedian(age_x, axis=0),
        summary=summary,
        path=BASE / "career_trajectory_archetypes_raw_elo.png",
    )

    agreement = (raw_membership["cluster"] == raw_membership["raw_elo_cluster"]).mean()
    print(summary.round(3).to_string(index=False))
    print(f"\nSame ordered cluster: {agreement:.1%}")
    print("\nRelative-Elo rows vs raw-Elo columns")
    print(pd.crosstab(raw_membership["cluster"], raw_membership["raw_elo_cluster"]).to_string())
    print("\nCluster-count selection")
    print(pd.DataFrame(selection_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
