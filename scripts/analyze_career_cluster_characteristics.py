from pathlib import Path

import numpy as np
import pandas as pd

from build_career_curve_clusters import GRID


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "analysis" / "4.3_career_curve_clusters" / "final_40_or_8_rule"
MEMBERSHIP = BASE / "player_career_trajectory_classification.csv"
UPDATES = ROOT / "results" / "final" / "final_elo_updates_canonical.csv.gz"
DIAGNOSTICS = ROOT / "results" / "final" / "final_elo_diagnostics_canonical.csv"
PLAYERS = ROOT / "processed_data" / "players.csv"


def main() -> None:
    membership = pd.read_csv(MEMBERSHIP)
    players = pd.read_csv(
        PLAYERS,
        usecols=["personId", "draftNumber", "draftRound", "birthDate"],
    ).rename(columns={"personId": "player_id"})
    players["birthDate"] = pd.to_datetime(players["birthDate"], errors="coerce")
    birth_map = players.set_index("player_id")["birthDate"]
    means = pd.read_csv(DIAGNOSTICS).set_index("season")["minutes_wmean"].to_dict()
    updates = pd.read_csv(
        UPDATES,
        usecols=["player_id", "game_date", "season", "rating_after"],
        parse_dates=["game_date"],
    ).sort_values(["player_id", "game_date"])
    updates = updates[updates["player_id"].isin(membership["player_id"])]
    updates["relative_elo"] = updates["rating_after"] - updates["season"].map(means)

    rows, curve_rows = [], []
    curves = {}
    for player_id, group in updates.groupby("player_id"):
        group = group.sort_values("game_date").reset_index(drop=True)
        elo21 = group["rating_after"].rolling(21, center=True, min_periods=1).mean()
        rel21 = group["relative_elo"].rolling(21, center=True, min_periods=1).mean()
        peak_idx = int(rel21.to_numpy().argmax())
        raw_peak_idx = int(group["rating_after"].to_numpy().argmax())
        raw_peak_age = (
            group.loc[raw_peak_idx, "game_date"] - birth_map.get(player_id, pd.NaT)
        ).days / 365.2425
        first_21 = float(group["rating_after"].head(21).mean())
        last_21 = float(group["rating_after"].tail(21).mean())
        peak_21 = float(elo21.iloc[peak_idx])
        progress = np.linspace(0.0, 1.0, len(group))
        sampled = np.interp(GRID, progress, rel21.to_numpy())
        standardized = (sampled - sampled.mean()) / sampled.std()
        curves[int(player_id)] = standardized
        rows.append(
            {
                "player_id": int(player_id),
                "career_mean_elo": group["rating_after"].mean(),
                "career_median_elo": group["rating_after"].median(),
                "first_21_game_mean_elo": first_21,
                "last_21_game_mean_elo": last_21,
                "peak_21_game_elo": peak_21,
                "raw_peak_elo": float(group.loc[raw_peak_idx, "rating_after"]),
                "raw_peak_date": group.loc[raw_peak_idx, "game_date"],
                "raw_peak_progress": float(raw_peak_idx / max(len(group) - 1, 1)),
                "raw_peak_age": float(raw_peak_age),
                "gain_first_to_peak": peak_21 - first_21,
                "decline_peak_to_last": peak_21 - last_21,
            }
        )
    detail = membership.merge(pd.DataFrame(rows), on="player_id")

    if "distance_to_cluster_center" not in detail.columns:
        for cluster, group in detail.groupby("cluster"):
            center = np.vstack([curves[int(i)] for i in group["player_id"]]).mean(axis=0)
            for player_id in group["player_id"]:
                curve_rows.append(
                    {
                        "player_id": int(player_id),
                        "distance_to_cluster_center": float(
                            np.sqrt(np.square(curves[int(player_id)] - center).sum())
                        ),
                    }
                )
        detail = detail.merge(pd.DataFrame(curve_rows), on="player_id")
    detail.to_csv(BASE / "cluster_characteristics_player_level.csv", index=False)

    summary_rows = []
    for cluster, group in detail.groupby("cluster"):
        row = {"cluster": cluster, "players": len(group)}
        for col in [
            "games", "seasons_observed", "career_mean_elo", "career_median_elo",
            "first_21_game_mean_elo", "peak_21_game_elo", "last_21_game_mean_elo",
            "raw_peak_elo", "raw_peak_progress", "raw_peak_age",
            "gain_first_to_peak", "decline_peak_to_last", "individual_peak_progress",
            "individual_peak_age", "draftNumber",
        ]:
            row[f"{col}_p25"] = group[col].quantile(0.25)
            row[f"{col}_median"] = group[col].median()
            row[f"{col}_p75"] = group[col].quantile(0.75)
        row["top14_share"] = group["draftNumber"].between(1, 14).mean()
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(BASE / "cluster_characteristics_summary.csv", index=False)

    representative = detail.sort_values(["cluster", "distance_to_cluster_center"]).groupby("cluster").head(15)
    representative.to_csv(BASE / "representative_players_by_curve_shape.csv", index=False)
    strongest = detail.sort_values(["cluster", "career_mean_elo"], ascending=[True, False]).groupby("cluster").head(15)
    strongest.to_csv(BASE / "strongest_players_by_cluster.csv", index=False)
    highest_raw_peak = detail.sort_values(
        ["cluster", "raw_peak_elo"], ascending=[True, False]
    ).groupby("cluster").head(15)
    highest_raw_peak.to_csv(BASE / "highest_raw_peak_players_by_cluster.csv", index=False)

    print(summary.round(2).to_string(index=False))
    print("\nLate-peak / sustained-rise representatives")
    print(
        representative[representative["cluster"] == 3][
            ["full_name", "games", "seasons_observed", "individual_peak_progress", "individual_peak_age", "career_mean_elo"]
        ].round(2).to_string(index=False)
    )
    print("\nStrongest late-peak / sustained-rise players")
    print(
        strongest[strongest["cluster"] == 3][
            ["full_name", "games", "individual_peak_progress", "individual_peak_age", "career_mean_elo", "peak_21_game_elo"]
        ].round(2).to_string(index=False)
    )


if __name__ == "__main__":
    main()
