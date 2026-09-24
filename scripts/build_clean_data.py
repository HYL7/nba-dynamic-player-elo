"""Generate the clean_data derived layer consumed by the Elo pipeline.

Reads processed_data/ (raw-ish CSV baseline) and writes clean_data/:
- games_clean.csv        availability flag for every game
- known_exemptions.csv   games excluded by the D-016 availability gate
- boxscores_clean.csv    played=1 rows with team-scaled minutes
- players_clean.csv      players with draft placeholder values normalized
- validation_report.md   reproducibility summary and consistency checks

Rules implemented: D-015 (minutes scaling), D-016 (availability gate),
D-017 (plus/minus availability), draft placeholder normalization.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("processed_data")
OUT = Path("clean_data")

TOLERANCE_MINUTES = 10.0
ON_COURT_START_SEASON = 1996  # season start year, i.e. 1996-97

KEEP_BOX_COLS = [
    "game_id",
    "game_date",
    "season",
    "season_id",
    "season_type",
    "track",
    "gameType",
    "gameLabel",
    "gameSubLabel",
    "seriesGameNumber",
    "personId",
    "firstName",
    "lastName",
    "playerteamId",
    "opponentteamId",
    "home",
    "win",
    "minutes",
    "scaled_minutes",
    "minutes_share",
    "plus_minus_available",
    "usable_game",
    "played",
    "points",
    "assists",
    "blocks",
    "steals",
    "fieldGoalsAttempted",
    "fieldGoalsMade",
    "fieldGoalsPercentage",
    "threePointersAttempted",
    "threePointersMade",
    "threePointersPercentage",
    "freeThrowsAttempted",
    "freeThrowsMade",
    "freeThrowsPercentage",
    "reboundsDefensive",
    "reboundsOffensive",
    "reboundsTotal",
    "foulsPersonal",
    "turnovers",
    "plusMinusPoints",
    "startingPosition",
    "comment",
]

BOX_READ_COLS = [
    "game_id",
    "game_date",
    "season",
    "season_id",
    "season_type",
    "track",
    "gameType",
    "gameLabel",
    "gameSubLabel",
    "seriesGameNumber",
    "personId",
    "firstName",
    "lastName",
    "playerteamId",
    "opponentteamId",
    "home",
    "win",
    "minutes",
    "played",
    "points",
    "assists",
    "blocks",
    "steals",
    "fieldGoalsAttempted",
    "fieldGoalsMade",
    "fieldGoalsPercentage",
    "threePointersAttempted",
    "threePointersMade",
    "threePointersPercentage",
    "freeThrowsAttempted",
    "freeThrowsMade",
    "freeThrowsPercentage",
    "reboundsDefensive",
    "reboundsOffensive",
    "reboundsTotal",
    "foulsPersonal",
    "turnovers",
    "plusMinusPoints",
    "startingPosition",
    "comment",
]


def reset_output_dir():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)


def read_games():
    return pd.read_csv(PROCESSED / "games.csv", low_memory=False)


def read_team_log():
    cols = [
        "game_id",
        "teamId",
        "opponentTeamId",
        "home",
        "win",
        "teamScore",
        "opponentScore",
        "numMinutes",
        "season",
        "season_id",
        "track",
        "season_type",
    ]
    return pd.read_csv(PROCESSED / "team_game_log.csv", usecols=cols, low_memory=False)


def build_team_game_stats():
    """Aggregate played minutes and row counts per game/team from boxscores."""
    chunks = []
    for chunk in pd.read_csv(
        PROCESSED / "boxscores.csv",
        usecols=["game_id", "playerteamId", "minutes", "played"],
        chunksize=200000,
    ):
        played = chunk.loc[chunk["played"] == 1]
        if played.empty:
            continue
        agg = (
            played.groupby(["game_id", "playerteamId"], as_index=False)
            .agg(played_minutes=("minutes", "sum"), played_rows=("minutes", "size"))
        )
        chunks.append(agg)
    stats = pd.concat(chunks, ignore_index=True)
    stats = (
        stats.groupby(["game_id", "playerteamId"], as_index=False)
        .agg(played_minutes=("played_minutes", "sum"), played_rows=("played_rows", "sum"))
    )
    stats = stats.rename(columns={"playerteamId": "teamId"})
    stats["game_id"] = stats["game_id"].astype("int64")
    stats["teamId"] = stats["teamId"].astype("int64")
    return stats


def classify_availability(team_stats, team_log):
    """Merge stats with the team log and apply the D-016 gate."""
    merged = team_log.merge(team_stats, on=["game_id", "teamId"], how="left")
    merged["played_rows"] = merged["played_rows"].fillna(0).astype(int)
    merged["diff_minutes"] = merged["played_minutes"] - merged["numMinutes"]

    reason = []
    for row in merged.itertuples(index=False):
        if row.played_rows == 0:
            reason.append("team_no_played_rows")
        elif abs(row.diff_minutes) > TOLERANCE_MINUTES:
            reason.append("minutes_out_of_tolerance")
        else:
            reason.append(None)
    merged["team_reason"] = reason

    per_game = (
        merged.groupby(["game_id", "season", "season_id", "track", "season_type"], as_index=False)
        .agg(
            team_count=("teamId", "count"),
            played_team_count=("played_rows", lambda s: (s > 0).sum()),
            team_reasons=("team_reason", lambda s: "|".join([x for x in s if isinstance(x, str)])),
            min_abs_diff=("diff_minutes", lambda s: np.min(np.abs(s.dropna())) if s.notna().any() else np.nan),
            max_abs_diff=("diff_minutes", lambda s: np.max(np.abs(s.dropna())) if s.notna().any() else np.nan),
            min_diff=("diff_minutes", "min"),
            max_diff=("diff_minutes", "max"),
            num_minutes_joined=("numMinutes", lambda s: s.notna().sum()),
        )
        .reset_index()
    )
    per_game["team_count"] = per_game["team_count"].astype(int)
    per_game["played_team_count"] = per_game["played_team_count"].astype(int)
    per_game["num_minutes_joined"] = per_game["num_minutes_joined"].astype(int)

    def game_reason(row):
        if row.team_count < 2:
            return "missing_team_rows"
        if row.played_team_count < 2:
            return "team_no_played_rows"
        if row.num_minutes_joined < 2:
            return "missing_num_minutes"
        if row.max_abs_diff > TOLERANCE_MINUTES:
            return "minutes_out_of_tolerance"
        return None

    per_game["reason"] = per_game.apply(game_reason, axis=1)
    per_game["is_usable"] = per_game["reason"].isna().astype(int)
    return merged, per_game


def write_games_and_exemptions(per_game):
    games = read_games()
    games = games.merge(
        per_game[
            [
                "game_id",
                "is_usable",
                "reason",
                "played_team_count",
                "min_diff",
                "max_diff",
                "team_reasons",
            ]
        ],
        on="game_id",
        how="left",
    )
    games["is_usable"] = games["is_usable"].fillna(0).astype(int)
    games["reason"] = games["reason"].fillna("missing_boxscore")
    games["played_team_count"] = games["played_team_count"].fillna(0).astype(int)
    games.to_csv(OUT / "games_clean.csv", index=False)

    exempt = per_game[per_game["is_usable"] == 0].copy()
    if not exempt.empty:
        exempt = exempt.rename(
            columns={
                "game_id": "game_id",
                "reason": "exemption_reason",
                "team_reasons": "team_level_reasons",
                "min_diff": "team_minutes_diff_min",
                "max_diff": "team_minutes_diff_max",
            }
        )
        exempt = exempt[
            [
                "game_id",
                "season",
                "season_id",
                "track",
                "season_type",
                "exemption_reason",
                "team_level_reasons",
                "team_minutes_diff_min",
                "team_minutes_diff_max",
                "played_team_count",
            ]
        ].sort_values(["season", "game_id"])
    else:
        exempt = per_game[["game_id"]].head(0)
    exempt.to_csv(OUT / "known_exemptions.csv", index=False)
    return games, exempt


def build_scale_map(team_stats, team_log):
    """Mapping from (game_id, teamId) to (scale, team_minutes)."""
    merged = team_log.merge(team_stats, on=["game_id", "teamId"], how="left")
    valid = merged[(merged["played_rows"].fillna(0) > 0) & (merged["numMinutes"].notna())].copy()
    valid["scale"] = valid["numMinutes"] / valid["played_minutes"]
    scale_map = {
        (int(row.game_id), int(row.teamId)): (float(row.scale), float(row.numMinutes))
        for row in valid.itertuples(index=False)
    }
    return scale_map


def write_clean_boxscores(scale_map, usable_game_ids):
    first = True
    total_rows = 0
    for chunk in pd.read_csv(PROCESSED / "boxscores.csv", usecols=BOX_READ_COLS, chunksize=200000, low_memory=False):
        chunk = chunk[chunk["played"] == 1].copy()
        if chunk.empty:
            continue
        chunk["game_id"] = chunk["game_id"].astype("int64")
        chunk["playerteamId"] = chunk["playerteamId"].astype("int64")
        keys = list(zip(chunk["game_id"], chunk["playerteamId"]))
        chunk["scale"] = [scale_map.get(k, (np.nan, np.nan))[0] for k in keys]
        chunk["team_minutes"] = [scale_map.get(k, (np.nan, np.nan))[1] for k in keys]
        chunk["scaled_minutes"] = chunk["minutes"] * chunk["scale"]
        chunk["minutes_share"] = chunk["scaled_minutes"] / chunk["team_minutes"]
        chunk["usable_game"] = chunk["game_id"].isin(usable_game_ids).astype(int)
        chunk["plus_minus_available"] = (
            (chunk["season"] >= ON_COURT_START_SEASON) & chunk["plusMinusPoints"].notna()
        ).astype(int)
        chunk = chunk.reindex(columns=KEEP_BOX_COLS)
        chunk = chunk.sort_values(["game_date", "game_id"])
        chunk.to_csv(OUT / "boxscores_clean.csv", mode="a", header=first, index=False)
        first = False
        total_rows += len(chunk)
    return total_rows


def write_clean_players():
    players = pd.read_csv(PROCESSED / "players.csv", low_memory=False)
    for col in ["draftYear", "draftRound", "draftNumber"]:
        players[col + "_raw"] = players[col]
        players[col] = players[col].replace(-1.0, np.nan).replace(-1, np.nan)
    players.to_csv(OUT / "players_clean.csv", index=False)
    return players


def check_references(games_clean, exempt, players_clean):
    issues = []
    team_log = read_team_log()
    teams = pd.read_csv(PROCESSED / "teams.csv", usecols=["teamId"], low_memory=False)
    team_ids = set(teams["teamId"].astype("int64"))
    game_ids = set(games_clean["game_id"])

    if team_log["game_id"].nunique() != len(game_ids):
        issues.append(f"team_game_log 覆盖 {team_log['game_id'].nunique()} 场，games.csv 有 {len(game_ids)} 场")
    dup_teams = team_log.groupby("game_id")["teamId"].size()
    bad = dup_teams[dup_teams != 2]
    if not bad.empty:
        issues.append(f"{len(bad)} 场比赛 team_game_log 行数不为 2")

    missing_team = set()
    for tid in pd.concat([team_log["teamId"], team_log["opponentTeamId"]]):
        try:
            if int(tid) not in team_ids:
                missing_team.add(int(tid))
        except (TypeError, ValueError):
            missing_team.add(tid)
    if missing_team:
        issues.append(f"team_game_log 中 {len(missing_team)} 个 teamId 不在 teams.csv: {sorted(missing_team)[:10]}")

    known_player_ids = set(players_clean["personId"].astype("int64"))
    box_player_ids = set()
    for chunk in pd.read_csv(OUT / "boxscores_clean.csv", usecols=["personId"], chunksize=200000):
        box_player_ids.update(chunk["personId"].astype("int64").tolist())
    missing_players = box_player_ids - known_player_ids
    if missing_players:
        issues.append(f"boxscores_clean 中 {len(missing_players)} 个 personId 不在 players_clean.csv")

    usable_ids = set(games_clean.loc[games_clean["is_usable"] == 1, "game_id"])
    clean_game_ids = set()
    for chunk in pd.read_csv(OUT / "boxscores_clean.csv", usecols=["game_id"], chunksize=200000):
        clean_game_ids.update(chunk["game_id"].astype("int64").tolist())
    exempt_ids = set(exempt["game_id"]) if not exempt.empty else set()
    overlapping = usable_ids & exempt_ids
    if overlapping:
        issues.append(f"{len(overlapping)} 场比赛同时被标记可用与豁免")

    # Score consistency: each game's two rows must agree on scores.
    score_check = team_log[["game_id", "teamId", "teamScore", "opponentScore"]].sort_values(["game_id", "teamId"])
    rows_per_game = score_check.groupby("game_id").size()
    two_row = rows_per_game[rows_per_game == 2].index
    score_check = score_check[score_check["game_id"].isin(two_row)]
    pivot = score_check.pivot(index="game_id", columns="teamId", values=["teamScore", "opponentScore"])
    mismatch = []
    for gid, row in pivot.iterrows():
        values = [x for x in row.values if pd.notna(x)]
        if len(values) == 4:
            s1, s2, o1, o2 = values
            if not (s1 == o2 and s2 == o1):
                mismatch.append(int(gid))
    if mismatch:
        issues.append(f"{len(mismatch)} 场比赛双方比分不一致: {mismatch[:10]}")

    return issues


def write_report(games_clean, exempt, total_rows, players_clean, issues):
    usable = int(games_clean["is_usable"].eq(1).sum())
    exempt_count = len(exempt)
    by_reason = exempt["exemption_reason"].value_counts() if not exempt.empty else pd.Series(dtype=int)
    by_season = exempt["season"].value_counts().sort_index() if not exempt.empty else pd.Series(dtype=int)

    plus_minus = pd.read_csv(
        OUT / "boxscores_clean.csv",
        usecols=["season", "plus_minus_available", "plusMinusPoints"],
    )
    pm = plus_minus.groupby(["season", "plus_minus_available"]).size().unstack(fill_value=0)
    pm = pm.rename(columns={0: "unavailable", 1: "available"})
    for col in ["available", "unavailable"]:
        if col not in pm.columns:
            pm[col] = 0
    pm["nonzero_plus_minus"] = plus_minus.groupby("season")["plusMinusPoints"].apply(lambda s: (s != 0).sum())

    draft = players_clean["draftNumber"]
    placeholder_count = int((players_clean["draftNumber_raw"].astype("Int64") == -1).fillna(False).sum())

    lines = [
        "# clean_data 校验报告",
        "",
        f"- 生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}",
        "- 输入：`processed_data/`；脚本：`build_clean_data.py`",
        "",
        "## 可用性 gate（D-016）",
        "",
        f"- 比赛总数：{len(games_clean)}",
        f"- 可用比赛：{usable}（{usable / len(games_clean):.2%}）",
        f"- 豁免比赛：{exempt_count}（{exempt_count / len(games_clean):.2%}）",
        "",
        "### 豁免原因",
        "",
    ]
    for reason, count in by_reason.items():
        lines.append(f"- `{reason}`：{count}")
    lines += ["", "### 豁免场次按赛季", "", "| season | 豁免场次 |", "|---|---|"]
    for season, count in by_season.items():
        lines.append(f"| {season} | {count} |")
    lines += [
        "",
        "## boxscores_clean",
        "",
        f"- `played=1` 行数：{total_rows}（含不可用比赛行；不可用场次保留原始行并按 `usable_game=0` 标记，Elo 只消费 `usable_game=1`）",
        "",
        "### 正负值可用性（D-017，按赛季）",
        "",
        "| season | available | unavailable | nonzero_plus_minus |",
        "|---|---|---|---|",
    ]
    for season, row in pm.iterrows():
        lines.append(f"| {season} | {row['available']} | {row['unavailable']} | {row['nonzero_plus_minus']} |")
    lines += [
        "",
        "## players_clean",
        "",
        f"- 球员总数：{len(players_clean)}",
        f"- 有 draftNumber：{int(draft.notna().sum())}",
        f"- 原 draftNumber=-1 占位：{placeholder_count}（已规范化为缺失，按落选秀处理）",
        "",
        "## 一致性检查",
        "",
    ]
    if issues:
        lines.append(f"- 发现 {len(issues)} 个问题：")
        for issue in issues:
            lines.append(f"  - {issue}")
    else:
        lines.append("- 未发现问题：比赛行数、比分一致性、球队/球员引用完整性均通过。")
    (OUT / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    reset_output_dir()
    print("reading games and team log")
    games = read_games()
    team_log = read_team_log()
    print("aggregating per-team boxscore minutes")
    team_stats = build_team_game_stats()
    print("classifying availability")
    _, per_game = classify_availability(team_stats, team_log)
    games_clean, exempt = write_games_and_exemptions(per_game)
    usable_ids = set(games_clean.loc[games_clean["is_usable"] == 1, "game_id"])
    print(f"usable games: {len(usable_ids)}, exempted: {len(exempt)}")

    print("building scale map and writing clean boxscores")
    scale_map = build_scale_map(team_stats, team_log)
    total_rows = write_clean_boxscores(scale_map, usable_ids)
    print(f"clean boxscore rows: {total_rows}")

    print("normalizing players")
    players_clean = write_clean_players()

    print("running consistency checks")
    issues = check_references(games_clean, exempt, players_clean)
    write_report(games_clean, exempt, total_rows, players_clean, issues)
    print("done")


if __name__ == "__main__":
    main()
