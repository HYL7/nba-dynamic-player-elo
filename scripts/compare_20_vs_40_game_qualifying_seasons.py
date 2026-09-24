from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "4.3_career_curve_clusters"
UPDATES = ROOT / "results" / "final" / "final_elo_updates_canonical.csv.gz"
MEMBERSHIP = OUT / "career_curve_membership_boundary_rules.csv"


def main() -> None:
    membership = pd.read_csv(MEMBERSHIP)
    updates = pd.read_csv(UPDATES, usecols=["player_id", "season"])
    season_games = updates.groupby(["player_id", "season"]).size().rename("games_in_season").reset_index()

    counts = []
    season_strings = []
    for player_id, group in season_games.groupby("player_id"):
        group = group.sort_values("season")
        q20 = group[group["games_in_season"] >= 20]
        q40 = group[group["games_in_season"] >= 40]
        counts.append(
            {
                "player_id": player_id,
                "seasons_20plus": len(q20),
                "seasons_40plus": len(q40),
                "seasons_20_to_39": int(group["games_in_season"].between(20, 39).sum()),
                "career_games_from_season_counts": int(group["games_in_season"].sum()),
                "career_seasons_from_season_counts": len(group),
                "median_games_per_season": float(group["games_in_season"].median()),
                "max_games_in_season": int(group["games_in_season"].max()),
            }
        )
        season_strings.append(
            {
                "player_id": player_id,
                "season_game_counts": "; ".join(
                    f"{int(r.season)}:{int(r.games_in_season)}" for r in group.itertuples()
                ),
            }
        )
    counts = pd.DataFrame(counts)
    season_strings = pd.DataFrame(season_strings)
    comparison = membership.merge(counts, on="player_id", how="left").merge(
        season_strings, on="player_id", how="left"
    )
    excluded = comparison[
        (comparison["seasons_20plus"] >= 5) & (comparison["seasons_40plus"] < 5)
    ].copy()
    excluded = excluded.sort_values(
        ["career_games_from_season_counts", "seasons_20plus"], ascending=False
    )
    excluded.to_csv(OUT / "players_in_20_game_but_not_40_game_rule.csv", index=False)

    print(f"excluded={len(excluded)}")
    print("cluster counts")
    print(excluded.groupby("cluster").size().to_string())
    print("summary")
    print(
        excluded[
            [
                "career_games_from_season_counts", "career_seasons_from_season_counts",
                "seasons_20plus", "seasons_40plus", "median_games_per_season",
                "max_games_in_season",
            ]
        ].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(1).to_string()
    )
    print("\nLongest excluded careers")
    print(
        excluded[
            [
                "full_name", "cluster", "career_games_from_season_counts",
                "career_seasons_from_season_counts", "seasons_20plus", "seasons_40plus",
                "median_games_per_season", "completed", "late_active",
            ]
        ].head(80).to_string(index=False)
    )


if __name__ == "__main__":
    main()
