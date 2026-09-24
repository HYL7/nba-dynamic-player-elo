"""Regenerate season display artifacts with season-end labels (D-024).

The engine and clean_data use the season's starting year internally
(1976 = 1976-77). Display CSVs label seasons by their ending year
(1998 = 1997-98) and keep both columns so they can be cross-referenced.

Usage:
    python build_results_summary.py
"""

import pandas as pd

from run_full_elo import load_usable_boxscores, season_metrics

RESULTS = "results"


def main():
    updates = pd.read_parquet(f"{RESULTS}/full_elo_updates_1976-2025.parquet")
    box = load_usable_boxscores(1976, 2025)

    metrics = season_metrics(updates, box)
    metrics["season_end"] = metrics["season"] + 1
    metrics["season_start"] = metrics["season"]
    metrics = metrics[[
        "season_end", "season_start", "n_players", "mean",
        "minutes_wmean", "sd", "mean_ge1000min", "top_rating",
    ]].sort_values("season_end")
    metrics.to_csv(f"{RESULTS}/full_elo_diagnostics_1976-2025.csv", index=False)

    players = pd.read_csv(
        "clean_data/players_clean.csv",
        usecols=["personId", "firstName", "lastName"],
        low_memory=False,
    )
    name_map = dict(
        zip(players["personId"], players["firstName"] + " " + players["lastName"])
    )
    last = updates.sort_values(["game_date", "game_id"]).groupby(
        ["season", "player_id"]
    ).tail(1)
    top = last.loc[last.groupby("season")["rating_after"].idxmax(), [
        "season", "player_id", "rating_after"
    ]].copy()
    top["player"] = top["player_id"].map(name_map)
    top["season_end"] = top["season"] + 1
    top["season_start"] = top["season"]
    top = top[[
        "season_end", "season_start", "player", "rating_after"
    ]].sort_values("season_end").reset_index(drop=True)
    top.to_csv(f"{RESULTS}/full_elo_top_by_season_1976-2025.csv", index=False)

    print(f"wrote {RESULTS}/full_elo_diagnostics_1976-2025.csv")
    print(f"wrote {RESULTS}/full_elo_top_by_season_1976-2025.csv")
    print(top.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
