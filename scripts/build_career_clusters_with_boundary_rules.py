from pathlib import Path

import numpy as np
import pandas as pd

from build_career_curve_clusters import GRID, kmeans, silhouette


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "4.3_career_curve_clusters"
UPDATES = ROOT / "results" / "final" / "final_elo_updates_canonical.csv.gz"
DIAGNOSTICS = ROOT / "results" / "final" / "final_elo_diagnostics_canonical.csv"
PLAYERS = ROOT / "processed_data" / "players.csv"

QUALIFYING_SEASON_GAMES = 20
MIN_QUALIFYING_SEASONS = 5
ACTIVE_MIN_AGE = 33.0


def make_curve(values: np.ndarray) -> np.ndarray | None:
    smooth = pd.Series(values).rolling(21, center=True, min_periods=1).mean().to_numpy()
    sampled = np.interp(GRID, np.linspace(0.0, 1.0, len(smooth)), smooth)
    if sampled.std() < 1e-9:
        return None
    return (sampled - sampled.mean()) / sampled.std()


def main() -> None:
    players = pd.read_csv(
        PLAYERS,
        usecols=["personId", "firstName", "lastName", "birthDate", "fromYear", "toYear"],
    ).rename(columns={"personId": "player_id"})
    players["birthDate"] = pd.to_datetime(players["birthDate"], errors="coerce")
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
        last_game=("game_date", "max"),
    ).reset_index().merge(players, on="player_id", how="left").set_index("player_id")
    meta["age_at_last"] = (meta["last_game"] - meta["birthDate"]).dt.days / 365.2425
    season_games = updates.groupby(["player_id", "season"]).size().rename("season_games").reset_index()
    qualifying = (
        season_games[season_games["season_games"] >= QUALIFYING_SEASON_GAMES]
        .groupby("player_id").size().rename("qualifying_seasons")
    )
    meta = meta.join(qualifying).fillna({"qualifying_seasons": 0})
    meta["full_start"] = (meta["first_season"] >= 1976) & (meta["first_season"] == meta["fromYear"])
    latest_season = int(updates["season"].max())
    meta["completed"] = meta["last_season"] < latest_season
    meta["late_active"] = (meta["last_season"] == latest_season) & (meta["age_at_last"] >= ACTIVE_MIN_AGE)
    keep = meta[
        meta["full_start"]
        & (meta["qualifying_seasons"] >= MIN_QUALIFYING_SEASONS)
        & (meta["completed"] | meta["late_active"])
    ].index

    ids, curves = [], []
    for player_id, group in updates[updates["player_id"].isin(keep)].groupby("player_id"):
        value = make_curve(group["relative_elo"].to_numpy())
        if value is not None:
            ids.append(int(player_id))
            curves.append(value)
    x = np.vstack(curves)
    labels, centers, inertia = kmeans(x, 3, seed=42, n_init=60)
    order = np.argsort([GRID[int(np.argmax(c))] for c in centers])
    centers = centers[order]
    remap = {int(old): int(new) + 1 for new, old in enumerate(order)}
    labels = np.asarray([remap[int(v)] for v in labels])

    membership = pd.DataFrame({"player_id": ids, "cluster": labels}).merge(
        meta.reset_index()[
            ["player_id", "full_name", "games", "seasons", "qualifying_seasons", "age_at_last", "completed", "late_active"]
        ],
        on="player_id",
        how="left",
    )
    membership.to_csv(OUT / "career_curve_membership_boundary_rules.csv", index=False)
    centers_out = []
    for cluster, center in enumerate(centers, start=1):
        row = {
            "cluster": cluster,
            "players": int((labels == cluster).sum()),
            "peak_progress": float(GRID[np.argmax(center)]),
        }
        row.update({f"p{int(p * 100):03d}": float(v) for p, v in zip(GRID, center)})
        centers_out.append(row)
    pd.DataFrame(centers_out).to_csv(OUT / "career_curve_centers_boundary_rules.csv", index=False)

    print(
        f"players={len(ids)} completed={membership['completed'].sum()} "
        f"late_active={membership['late_active'].sum()} silhouette={silhouette(x, labels):.4f}"
    )
    print(pd.DataFrame(centers_out)[["cluster", "players", "peak_progress"]].to_string(index=False))
    print("\nActive players by age")
    print(
        membership[membership["late_active"]]
        .assign(age=lambda d: d["age_at_last"].round(0).astype(int))
        .groupby("age").size().to_string()
    )


if __name__ == "__main__":
    main()
