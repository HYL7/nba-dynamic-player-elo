from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "analysis" / "4.3_career_curve_clusters" / "final_40_or_8_rule"
CHARACTERISTICS = BASE / "cluster_characteristics_player_level_raw_elo.csv"
UPDATES = ROOT / "results" / "final" / "final_elo_updates_canonical.csv.gz"
OUTPUT = BASE / "peak_elo_1800_1999_contemporaneous_rank.csv"
SUMMARY_OUTPUT = BASE / "peak_elo_1800_1999_rank_summary.csv"
ALL_OUTPUT = BASE / "all_players_peak_elo_contemporaneous_rank.csv"
LEVEL_SUMMARY_OUTPUT = BASE / "peak_elo_level_contemporaneous_rank_summary.csv"


def main() -> None:
    players = pd.read_csv(CHARACTERISTICS)
    targets = players[
        ["player_id", "full_name", "raw_peak_elo", "raw_peak_date", "raw_elo_cluster"]
    ].copy()
    targets["raw_peak_date"] = pd.to_datetime(targets["raw_peak_date"])

    updates = pd.read_csv(
        UPDATES,
        usecols=["player_id", "game_date", "season", "rating_after"],
        parse_dates=["game_date"],
    ).sort_values(["game_date", "player_id"])

    peak_events = updates.merge(
        targets[["player_id", "raw_peak_date"]],
        left_on=["player_id", "game_date"],
        right_on=["player_id", "raw_peak_date"],
        how="inner",
    )
    peak_events = peak_events.loc[
        (peak_events["rating_after"] - peak_events.groupby("player_id")["rating_after"].transform("max")).abs() < 1e-8
    ].drop_duplicates("player_id")
    target_season = peak_events.set_index("player_id")["season"].to_dict()

    target_by_date = {
        date: group["player_id"].astype(int).tolist()
        for date, group in targets.groupby("raw_peak_date")
    }
    current_rating: dict[int, float] = {}
    active_by_season: dict[int, set[int]] = {}
    results = []

    for game_date, day in updates.groupby("game_date", sort=True):
        for row in day.itertuples(index=False):
            player_id = int(row.player_id)
            season = int(row.season)
            current_rating[player_id] = float(row.rating_after)
            active_by_season.setdefault(season, set()).add(player_id)

        if game_date not in target_by_date:
            continue
        for player_id in target_by_date[game_date]:
            season = int(target_season[player_id])
            active = active_by_season.get(season, set())
            ratings = sorted(
                ((pid, current_rating[pid]) for pid in active if pid in current_rating),
                key=lambda item: (-item[1], item[0]),
            )
            rank = next(index for index, (pid, _) in enumerate(ratings, start=1) if pid == player_id)
            results.append(
                {
                    "player_id": player_id,
                    "peak_date": game_date,
                    "season": season,
                    "contemporaneous_rank": rank,
                    "active_players_ranked": len(ratings),
                }
            )

    ranked = targets.merge(pd.DataFrame(results), on="player_id", how="left")
    ranked["peak_elo_band"] = pd.cut(
        ranked["raw_peak_elo"],
        [-np.inf, 1600, 1800, 2000, np.inf],
        right=False,
        labels=["Below 1600", "1600–1799", "1800–1999", "2000+"],
    )
    ranked = ranked.sort_values(["raw_peak_elo", "full_name"], ascending=[False, True])
    ranked.to_csv(ALL_OUTPUT, index=False)
    ranked[ranked["raw_peak_elo"].ge(1800) & ranked["raw_peak_elo"].lt(2000)].to_csv(
        OUTPUT, index=False
    )

    summary_rows = []
    for band, group in ranked.groupby("peak_elo_band", observed=True):
        summary_rows.append(
            {
                "peak_elo_band": band,
                "players": len(group),
                "rank_p25": group["contemporaneous_rank"].quantile(0.25),
                "rank_median": group["contemporaneous_rank"].median(),
                "rank_p75": group["contemporaneous_rank"].quantile(0.75),
                "rank_min": group["contemporaneous_rank"].min(),
                "rank_max": group["contemporaneous_rank"].max(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(LEVEL_SUMMARY_OUTPUT, index=False)

    upper = ranked[ranked["raw_peak_elo"].ge(1800) & ranked["raw_peak_elo"].lt(2000)].copy()
    upper["peak_elo_band"] = pd.cut(
        upper["raw_peak_elo"],
        [1800, 1850, 1900, 1950, 2000],
        right=False,
        labels=["1800–1849", "1850–1899", "1900–1949", "1950–1999"],
    )
    detailed_rows = []
    for band, group in upper.groupby("peak_elo_band", observed=True):
        detailed_rows.append(
            {
                "peak_elo_band": band,
                "players": len(group),
                "rank_p25": group["contemporaneous_rank"].quantile(0.25),
                "rank_median": group["contemporaneous_rank"].median(),
                "rank_p75": group["contemporaneous_rank"].quantile(0.75),
                "rank_min": group["contemporaneous_rank"].min(),
                "rank_max": group["contemporaneous_rank"].max(),
            }
        )
    pd.DataFrame(detailed_rows).to_csv(SUMMARY_OUTPUT, index=False)

    print(f"Players ranked: {ranked['contemporaneous_rank'].notna().sum()} / {len(ranked)}")
    print(summary.round(1).to_string(index=False))
    print("\nOverall rank quantiles")
    print(ranked["contemporaneous_rank"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_string())


if __name__ == "__main__":
    main()
