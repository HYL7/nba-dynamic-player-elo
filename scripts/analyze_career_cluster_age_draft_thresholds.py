from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "4.3_career_curve_clusters"
UPDATES = ROOT / "results" / "final" / "final_elo_updates_canonical.csv.gz"
DIAGNOSTICS = ROOT / "results" / "final" / "final_elo_diagnostics_canonical.csv"
PLAYERS = ROOT / "processed_data" / "players.csv"
MEMBERSHIP = OUT / "career_curve_cluster_membership.csv"
CENTERS = OUT / "career_curve_cluster_centers.csv"
GRID = np.linspace(0.0, 1.0, 41)


def curve_from_values(values: np.ndarray) -> np.ndarray | None:
    if len(values) < 2:
        return None
    smooth = pd.Series(values).rolling(21, center=True, min_periods=1).mean().to_numpy()
    curve = np.interp(GRID, np.linspace(0.0, 1.0, len(smooth)), smooth)
    sd = curve.std()
    if sd < 1e-9:
        return None
    return (curve - curve.mean()) / sd


def assign(curve: np.ndarray, centers: np.ndarray) -> int:
    return int(np.square(centers - curve).sum(axis=1).argmin()) + 1


def balanced_accuracy(truth: np.ndarray, pred: np.ndarray) -> float:
    recalls = [(pred[truth == k] == k).mean() for k in sorted(np.unique(truth))]
    return float(np.mean(recalls))


def main() -> None:
    membership = pd.read_csv(MEMBERSHIP)
    players = pd.read_csv(
        PLAYERS,
        usecols=[
            "personId", "birthDate", "draftYear", "draftRound", "draftNumber",
        ],
    ).rename(columns={"personId": "player_id"})
    players["birthDate"] = pd.to_datetime(players["birthDate"], errors="coerce")
    membership = membership.merge(players, on="player_id", how="left")

    diagnostics = pd.read_csv(DIAGNOSTICS).set_index("season")
    season_mean = diagnostics["minutes_wmean"].to_dict()
    updates = pd.read_csv(
        UPDATES,
        usecols=["player_id", "game_date", "season", "rating_after"],
        parse_dates=["game_date"],
    ).sort_values(["player_id", "game_date"])
    updates = updates[updates["player_id"].isin(membership["player_id"])]
    updates["relative_elo"] = updates["rating_after"] - updates["season"].map(season_mean)
    birth_map = membership.set_index("player_id")["birthDate"]
    updates["birthDate"] = updates["player_id"].map(birth_map)
    updates["age"] = (updates["game_date"] - updates["birthDate"]).dt.days / 365.2425

    # Map normalized career progress to actual ages.
    age_rows = []
    player_summaries = []
    progress_targets = [0.0, 0.18, 0.25, 0.43, 0.50, 0.62, 0.75, 1.0]
    member_index = membership.set_index("player_id")
    histories = {}
    for player_id, group in updates.groupby("player_id", sort=False):
        group = group.dropna(subset=["age", "relative_elo"]).reset_index(drop=True)
        if group.empty:
            continue
        histories[int(player_id)] = group
        n = len(group)
        for progress in progress_targets:
            idx = int(round(progress * (n - 1)))
            age_rows.append(
                {"player_id": player_id, "progress": progress, "age": group.loc[idx, "age"]}
            )
        peak_progress = float(member_index.loc[player_id, "individual_peak_progress"])
        peak_idx = int(round(peak_progress * (n - 1)))
        player_summaries.append(
            {
                "player_id": player_id,
                "start_age": group.loc[0, "age"],
                "peak_age": group.loc[peak_idx, "age"],
                "end_age": group.loc[n - 1, "age"],
            }
        )

    age_map = pd.DataFrame(age_rows)
    age_summary = age_map.groupby("progress")["age"].agg(
        n="size",
        p10=lambda x: x.quantile(0.10),
        p25=lambda x: x.quantile(0.25),
        median="median",
        p75=lambda x: x.quantile(0.75),
        p90=lambda x: x.quantile(0.90),
    ).reset_index()
    age_summary.to_csv(OUT / "career_progress_to_age.csv", index=False)

    summary = membership.merge(pd.DataFrame(player_summaries), on="player_id", how="left")
    summary["draft_group"] = np.select(
        [
            summary["draftNumber"].between(1, 14),
            summary["draftRound"].eq(1) & summary["draftNumber"].gt(14),
            summary["draftRound"].gt(1),
        ],
        ["Top 14", "Other first round", "Later rounds"],
        default="Undrafted/unknown",
    )
    cluster_summary = summary.groupby("cluster").agg(
        players=("player_id", "size"),
        debut_age_median=("start_age", "median"),
        peak_age_median=("peak_age", "median"),
        final_age_median=("end_age", "median"),
        career_games_median=("games", "median"),
        draft_pick_median=("draftNumber", "median"),
        draft_pick_mean=("draftNumber", "mean"),
    ).reset_index()
    draft_shares = pd.crosstab(summary["cluster"], summary["draft_group"], normalize="index").reset_index()
    cluster_summary.to_csv(OUT / "career_cluster_age_draft_summary.csv", index=False)
    draft_shares.to_csv(OUT / "career_cluster_draft_group_shares.csv", index=False)

    center_df = pd.read_csv(CENTERS).sort_values("cluster")
    center_cols = [f"p{int(p * 100):03d}" for p in GRID]
    centers = center_df[center_cols].to_numpy()
    truth_map = membership.set_index("player_id")["cluster"].astype(int).to_dict()

    # Completed-player censoring test by age: retain only games observed by the cutoff age.
    # The at-risk result excludes careers already complete at the cutoff and requires at
    # least one subsequent year, avoiding inflated accuracy from already-retired players.
    age_tests = []
    for cutoff_age in np.arange(29.0, 39.5, 0.5):
        for min_games in [400, 500, 600, 700, 800, 900]:
            truth, pred = [], []
            risk_truth, risk_pred = [], []
            for player_id, group in histories.items():
                partial = group[group["age"] <= cutoff_age]
                if len(partial) < min_games:
                    continue
                curve = curve_from_values(partial["relative_elo"].to_numpy())
                if curve is None:
                    continue
                truth.append(truth_map[player_id])
                pred.append(assign(curve, centers))
                if group["age"].max() >= cutoff_age + 1.0:
                    risk_truth.append(truth_map[player_id])
                    risk_pred.append(assign(curve, centers))
            if not truth:
                continue
            truth_arr, pred_arr = np.asarray(truth), np.asarray(pred)
            risk_truth_arr, risk_pred_arr = np.asarray(risk_truth), np.asarray(risk_pred)
            age_tests.append(
                {
                    "cutoff_age": cutoff_age,
                    "min_games": min_games,
                    "players": len(truth_arr),
                    "accuracy": (truth_arr == pred_arr).mean(),
                    "balanced_accuracy": balanced_accuracy(truth_arr, pred_arr),
                    "at_risk_players": len(risk_truth_arr),
                    "at_risk_accuracy": (
                        (risk_truth_arr == risk_pred_arr).mean() if len(risk_truth_arr) else np.nan
                    ),
                    "at_risk_balanced_accuracy": (
                        balanced_accuracy(risk_truth_arr, risk_pred_arr)
                        if len(risk_truth_arr) and len(np.unique(risk_truth_arr)) == 3
                        else np.nan
                    ),
                }
            )
    age_test_df = pd.DataFrame(age_tests)
    age_test_df.to_csv(OUT / "career_cluster_censoring_by_age_and_games.csv", index=False)

    # Game-count-only censoring test.
    game_tests = []
    for cutoff_games in range(400, 1201, 50):
        truth, pred = [], []
        for player_id, group in histories.items():
            if len(group) < cutoff_games:
                continue
            curve = curve_from_values(group.iloc[:cutoff_games]["relative_elo"].to_numpy())
            if curve is None:
                continue
            truth.append(truth_map[player_id])
            pred.append(assign(curve, centers))
        truth_arr, pred_arr = np.asarray(truth), np.asarray(pred)
        game_tests.append(
            {
                "cutoff_games": cutoff_games,
                "players": len(truth_arr),
                "accuracy": (truth_arr == pred_arr).mean(),
                "balanced_accuracy": balanced_accuracy(truth_arr, pred_arr),
            }
        )
    game_test_df = pd.DataFrame(game_tests)
    game_test_df.to_csv(OUT / "career_cluster_censoring_by_games.csv", index=False)

    # Oracle experiment: how much of the eventual career must be observed? This cannot
    # be applied directly to active players, but establishes the underlying requirement.
    progress_tests = []
    for observed_share in np.arange(0.50, 1.001, 0.05):
        truth, pred = [], []
        for player_id, group in histories.items():
            cutoff = max(2, int(np.floor(len(group) * observed_share)))
            curve = curve_from_values(group.iloc[:cutoff]["relative_elo"].to_numpy())
            if curve is None:
                continue
            truth.append(truth_map[player_id])
            pred.append(assign(curve, centers))
        truth_arr, pred_arr = np.asarray(truth), np.asarray(pred)
        progress_tests.append(
            {
                "observed_career_share": observed_share,
                "players": len(truth_arr),
                "accuracy": (truth_arr == pred_arr).mean(),
                "balanced_accuracy": balanced_accuracy(truth_arr, pred_arr),
            }
        )
    progress_test_df = pd.DataFrame(progress_tests)
    progress_test_df.to_csv(OUT / "career_cluster_censoring_by_observed_share.csv", index=False)

    print("CAREER PROGRESS TO AGE")
    print(age_summary.round(2).to_string(index=False))
    print("\nCLUSTER AGE AND DRAFT SUMMARY")
    print(cluster_summary.round(2).to_string(index=False))
    print("\nDRAFT GROUP SHARES")
    print(draft_shares.round(3).to_string(index=False))
    print("\nAGE/GAME CENSORING: AGE 32+")
    print(
        age_test_df[(age_test_df["cutoff_age"] >= 32) & (age_test_df["min_games"].isin([600, 700, 800]))]
        .round(3).to_string(index=False)
    )
    print("\nGAME CENSORING")
    print(game_test_df[game_test_df["cutoff_games"].isin([600, 700, 800, 900, 1000, 1100])].round(3).to_string(index=False))
    print("\nOBSERVED CAREER SHARE CENSORING")
    print(progress_test_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
