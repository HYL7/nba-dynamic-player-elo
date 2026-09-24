from pathlib import Path

import numpy as np
import pandas as pd

from build_career_curve_clusters import GRID, kmeans, silhouette


ROOT = Path(__file__).resolve().parents[1]
BASE_OUT = ROOT / "results" / "analysis" / "4.3_career_curve_clusters" / "final_40_game_rule"
OUT = ROOT / "results" / "analysis" / "4.3_career_curve_clusters" / "rule_comparison_40_or_8"
UPDATES = ROOT / "results" / "final" / "final_elo_updates_canonical.csv.gz"
DIAGNOSTICS = ROOT / "results" / "final" / "final_elo_diagnostics_canonical.csv"
PLAYERS = ROOT / "processed_data" / "players.csv"


def make_curve(values: np.ndarray) -> np.ndarray | None:
    smooth = pd.Series(values).rolling(21, center=True, min_periods=1).mean().to_numpy()
    sampled = np.interp(GRID, np.linspace(0.0, 1.0, len(smooth)), smooth)
    if sampled.std() < 1e-9:
        return None
    return (sampled - sampled.mean()) / sampled.std()


def summarize_levels(membership: pd.DataFrame, updates: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    player_levels = updates[updates["player_id"].isin(membership["player_id"])].groupby("player_id").agg(
        career_mean_elo=("rating_after", "mean"),
        career_median_elo=("rating_after", "median"),
        career_mean_relative_elo=("relative_elo", "mean"),
    ).reset_index()
    data = membership.merge(player_levels, on="player_id", how="left")
    rows = []
    for cluster, group in data.groupby("cluster"):
        row = {"sample_rule": label, "cluster": cluster, "players": len(group)}
        for col in [
            "career_mean_elo", "career_median_elo", "career_mean_relative_elo",
            "individual_peak_elo_21g", "individual_peak_relative_elo_21g",
        ]:
            if col not in group:
                continue
            row[f"{col}_p25"] = group[col].quantile(0.25)
            row[f"{col}_median"] = group[col].median()
            row[f"{col}_p75"] = group[col].quantile(0.75)
        rows.append(row)
    level_summary = pd.DataFrame(rows)

    data["debut_era"] = pd.cut(
        data["first_season"],
        bins=[1975, 1979, 1989, 1999, 2009, 2019, 2029],
        labels=["1976-79", "1980s", "1990s", "2000s", "2010s", "2020s"],
    )
    era = pd.crosstab(data["debut_era"], data["cluster"], margins=False)
    era_share = era.div(era.sum(axis=1), axis=0)
    era_out = era.reset_index().melt(id_vars="debut_era", var_name="cluster", value_name="players")
    share_out = era_share.reset_index().melt(id_vars="debut_era", var_name="cluster", value_name="share_within_era")
    era_out = era_out.merge(share_out, on=["debut_era", "cluster"])
    era_out.insert(0, "sample_rule", label)
    return level_summary, era_out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(BASE_OUT / "player_career_trajectory_classification.csv")
    players = pd.read_csv(
        PLAYERS,
        usecols=["personId", "firstName", "lastName", "birthDate", "fromYear"],
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
    latest = int(updates["season"].max())
    meta = updates.groupby("player_id").agg(
        games=("rating_after", "size"),
        seasons_observed=("season", "nunique"),
        first_season=("season", "min"),
        last_season=("season", "max"),
        first_game=("game_date", "min"),
        last_game=("game_date", "max"),
    ).reset_index().merge(players, on="player_id", how="left").set_index("player_id")
    meta["age_at_last"] = (meta["last_game"] - meta["birthDate"]).dt.days / 365.2425
    season_games = updates.groupby(["player_id", "season"]).size().rename("season_games").reset_index()
    q40 = season_games[season_games["season_games"] >= 40].groupby("player_id").size().rename("seasons_40plus")
    meta = meta.join(q40).fillna({"seasons_40plus": 0})
    meta["full_start"] = (meta["first_season"] >= 1976) & (meta["first_season"] == meta["fromYear"])
    meta["appeared_latest"] = meta["last_season"] == latest
    meta["provisional_active"] = meta["appeared_latest"] & (meta["age_at_last"] >= 33.0)
    meta["eligible_end"] = (~meta["appeared_latest"]) | meta["provisional_active"]
    eligible = meta[
        meta["full_start"]
        & meta["eligible_end"]
        & ((meta["seasons_40plus"] >= 5) | (meta["seasons_observed"] >= 8))
    ]

    curves, ids = [], []
    peak_rows = []
    for player_id, group in updates[updates["player_id"].isin(eligible.index)].groupby("player_id"):
        group = group.sort_values("game_date").reset_index(drop=True)
        smoothed_relative = group["relative_elo"].rolling(21, center=True, min_periods=1).mean().to_numpy()
        smoothed_elo = group["rating_after"].rolling(21, center=True, min_periods=1).mean().to_numpy()
        value = make_curve(group["relative_elo"].to_numpy())
        if value is None:
            continue
        peak_idx = int(np.argmax(smoothed_relative))
        progress = peak_idx / max(len(group) - 1, 1)
        birth = eligible.loc[player_id, "birthDate"]
        peak_age = (group.loc[peak_idx, "game_date"] - birth).days / 365.2425
        ids.append(int(player_id))
        curves.append(value)
        peak_rows.append(
            {
                "player_id": int(player_id),
                "individual_peak_progress": progress,
                "individual_peak_elo_21g": smoothed_elo[peak_idx],
                "individual_peak_relative_elo_21g": smoothed_relative[peak_idx],
                "individual_peak_age": peak_age,
            }
        )
    x = np.vstack(curves)
    labels, centers, _ = kmeans(x, 3, seed=42, n_init=80)
    order = np.argsort([GRID[int(np.argmax(c))] for c in centers])
    centers = centers[order]
    remap = {int(old): int(new) + 1 for new, old in enumerate(order)}
    labels = np.asarray([remap[int(v)] for v in labels])

    membership = pd.DataFrame({"player_id": ids, "cluster": labels}).merge(
        pd.DataFrame(peak_rows), on="player_id"
    ).merge(meta.reset_index(), on="player_id")
    membership.to_csv(OUT / "player_classification_40x5_or_8_seasons.csv", index=False)
    added = membership[~membership["player_id"].isin(baseline["player_id"])].copy()
    added.to_csv(OUT / "players_added_by_8_season_alternative.csv", index=False)
    shared = membership[membership["player_id"].isin(baseline["player_id"])][["player_id", "cluster"]]
    shared = shared.merge(baseline[["player_id", "cluster"]], on="player_id", suffixes=("_or", "_base"))

    baseline_levels, baseline_era = summarize_levels(baseline, updates, "five_40_game_seasons")
    or_levels, or_era = summarize_levels(membership, updates, "five_40_game_seasons_or_eight_total")
    pd.concat([baseline_levels, or_levels], ignore_index=True).to_csv(OUT / "cluster_elo_level_summary.csv", index=False)
    pd.concat([baseline_era, or_era], ignore_index=True).to_csv(OUT / "cluster_debut_era_distribution.csv", index=False)

    print(
        f"baseline={len(baseline)} or_rule={len(membership)} added={len(added)} "
        f"shared_stability={(shared['cluster_or'] == shared['cluster_base']).mean():.4f} "
        f"silhouette={silhouette(x, labels):.4f}"
    )
    print("cluster sizes and peaks")
    for cluster in range(1, 4):
        print(cluster, int((labels == cluster).sum()), float(GRID[np.argmax(centers[cluster - 1])]))
    print("\nAdded-player summary")
    print(added[["games", "seasons_observed", "seasons_40plus", "age_at_last"]].describe().round(1).to_string())
    print("\nLongest added players")
    print(added.sort_values("games", ascending=False)[["full_name", "games", "seasons_observed", "seasons_40plus", "cluster"]].head(50).to_string(index=False))
    print("\nBaseline Elo level summary")
    print(baseline_levels.round(2).to_string(index=False))
    print("\nBaseline era distribution")
    print(baseline_era.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
