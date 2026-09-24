import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("raw_data")
OUT = Path("processed_data")
OUT.mkdir(exist_ok=True)

START_SEASON = 1976
END_SEASON = 2025

GAMES_FILE = RAW / "Data2" / "Games.csv"
PLAYER_STATS_FILE = RAW / "Data2" / "PlayerStatistics.csv"
PLAYER_STATS_EXTENDED_FILE = RAW / "Data2" / "PlayerStatisticsExtended.csv"
TEAM_STATS_FILE = RAW / "Data2" / "TeamStatistics.csv"
TEAM_STATS_EXTENDED_FILE = RAW / "Data2" / "TeamStatisticsExtended.csv"
PLAYERS_FILE = RAW / "Data2" / "Players.csv"
TEAM_HISTORIES_FILE = RAW / "Data2" / "TeamHistories.csv"

DATA1_CSV = RAW / "Data1" / "csv"
COMMON_PLAYER_INFO_FILE = DATA1_CSV / "common_player_info.csv"
DRAFT_HISTORY_FILE = DATA1_CSV / "draft_history.csv"
INACTIVE_PLAYERS_FILE = DATA1_CSV / "inactive_players.csv"
TEAM_HISTORY_FILE = DATA1_CSV / "team_history.csv"


def parse_minutes(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    if ":" in text:
        parts = text.split(":")
        try:
            minutes = float(parts[0]) + float(parts[1]) / 60.0
            return minutes if minutes >= 0 else np.nan
        except ValueError:
            return np.nan
    try:
        minutes = float(text)
    except ValueError:
        return np.nan
    return minutes if minutes >= 0 else np.nan


def assign_season(dates):
    """Assign NBA season start year and label from a datetime Series."""
    dates = pd.to_datetime(dates, errors="coerce")
    year = dates.dt.year
    month = dates.dt.month
    season_start = np.where(month < 7, year - 1, year)
    # The 2019-20 bubble was played in July/August 2020 but belongs to 2019-20.
    bubble = (year == 2020) & (month >= 7) & (month <= 9)
    season_start = np.where(bubble, 2019, season_start)
    season_start = pd.Series(season_start, index=dates.index, dtype="Int64")
    season_label = season_start.astype("Int64").astype(str) + "-" + (season_start.astype("Int64") + 1).astype(str).str[-2:]
    return season_start, season_label


def classify_game(game_type, game_sub_label):
    game_type = game_type.fillna("").astype(str).str.strip().str.lower()
    sub = game_sub_label.fillna("").astype(str).str.strip().str.lower()
    is_championship = sub.eq("championship")
    is_regular = game_type.isin(["regular season", "nba emirates cup"]) & ~is_championship
    is_playoffs = game_type.isin(["playoffs", "play-in tournament"])
    track = np.select(
        [is_regular, is_playoffs],
        ["regular", "playoffs"],
        default="excluded",
    )
    season_type = np.select(
        [is_regular, is_playoffs],
        ["Regular Season", "Playoffs"],
        default="Excluded",
    )
    return pd.Series(track, index=game_type.index), pd.Series(season_type, index=game_type.index)


def normalize_team_key(city, name):
    city = city.fillna("").astype(str).str.strip().str.lower()
    name = name.fillna("").astype(str).str.strip().str.lower()
    return city + "|" + name


def build_team_mapping():
    frames = []
    ts = pd.read_csv(
        TEAM_STATS_FILE,
        low_memory=False,
        usecols=["teamId", "teamCity", "teamName", "opponentTeamId", "opponentTeamCity", "opponentTeamName"],
    )
    frames.append(ts[["teamId", "teamCity", "teamName"]].rename(columns={"teamId": "team_id", "teamCity": "city", "teamName": "name"}))
    frames.append(ts[["opponentTeamId", "opponentTeamCity", "opponentTeamName"]].rename(columns={"opponentTeamId": "team_id", "opponentTeamCity": "city", "opponentTeamName": "name"}))

    games = pd.read_csv(
        GAMES_FILE,
        low_memory=False,
        usecols=["hometeamId", "hometeamCity", "hometeamName", "awayteamId", "awayteamCity", "awayteamName"],
    )
    frames.append(games[["hometeamId", "hometeamCity", "hometeamName"]].rename(columns={"hometeamId": "team_id", "hometeamCity": "city", "hometeamName": "name"}))
    frames.append(games[["awayteamId", "awayteamCity", "awayteamName"]].rename(columns={"awayteamId": "team_id", "awayteamCity": "city", "awayteamName": "name"}))

    hist = pd.read_csv(
        TEAM_HISTORIES_FILE,
        low_memory=False,
        usecols=["teamId", "teamCity", "teamName"],
    )
    frames.append(hist.rename(columns={"teamId": "team_id", "teamCity": "city", "teamName": "name"}))

    mapping = pd.concat(frames, ignore_index=True)
    mapping = mapping.dropna(subset=["team_id", "name"])
    mapping = mapping.drop_duplicates(subset=["city", "name"], keep="first")
    mapping["key"] = normalize_team_key(mapping["city"], mapping["name"])
    mapping = mapping.drop_duplicates(subset=["key"], keep="first")
    return mapping.set_index("key")["team_id"].to_dict()


def fill_team_ids(df, team_map, player_cols, opponent_cols):
    player_key = normalize_team_key(df[player_cols[0]], df[player_cols[1]])
    opponent_key = normalize_team_key(df[opponent_cols[0]], df[opponent_cols[1]])
    if "playerteamId" in df.columns:
        df["playerteamId"] = df["playerteamId"].fillna(player_key.map(team_map))
    if "opponentteamId" in df.columns:
        df["opponentteamId"] = df["opponentteamId"].fillna(opponent_key.map(team_map))
    return df


def read_selected_game_ids():
    games = pd.read_csv(
        GAMES_FILE,
        low_memory=False,
        usecols=["gameId", "gameDateTimeEst", "gameType", "gameSubLabel"],
    )
    games["game_date"] = pd.to_datetime(games["gameDateTimeEst"], errors="coerce")
    season_start, season_label = assign_season(games["game_date"])
    games["season"] = season_start
    games["season_id"] = season_label
    track, season_type = classify_game(games["gameType"], games["gameSubLabel"])
    games["track"] = track
    games["season_type"] = season_type
    selected = games[
        (games["season"].between(START_SEASON, END_SEASON))
        & (games["track"] != "excluded")
    ].copy()
    return selected[["gameId", "game_date", "season", "season_id", "track", "season_type"]].rename(columns={"gameId": "game_id"})


def write_games(selected_games, team_map):
    games = pd.read_csv(GAMES_FILE, low_memory=False)
    games = games.rename(columns={"gameId": "game_id"})
    games = games.merge(selected_games[["game_id", "season", "season_id", "track", "season_type"]], on="game_id", how="inner")
    games["game_date"] = pd.to_datetime(games["gameDateTimeEst"], errors="coerce")
    output_cols = [
        "game_id",
        "game_date",
        "season",
        "season_id",
        "season_type",
        "track",
        "gameType",
        "gameSubtype",
        "gameLabel",
        "gameSubLabel",
        "seriesGameNumber",
        "hometeamCity",
        "hometeamName",
        "hometeamId",
        "awayteamCity",
        "awayteamName",
        "awayteamId",
        "homeScore",
        "awayScore",
        "winner",
        "attendance",
        "arenaId",
        "arenaName",
        "arenaCity",
        "arenaState",
        "officials",
        "gameDate",
    ]
    games = games.reindex(columns=output_cols)
    games = games.sort_values(["game_date", "game_id"])
    games.to_csv(OUT / "games.csv", index=False)
    return games


def write_team_game_log(selected_games):
    chunks = pd.read_csv(TEAM_STATS_FILE, low_memory=False, chunksize=200000)
    selected_ids = set(selected_games["game_id"])
    first = True
    for chunk in chunks:
        chunk = chunk.rename(columns={"gameId": "game_id"})
        chunk = chunk[chunk["game_id"].isin(selected_ids)]
        chunk = chunk[(chunk["teamId"].fillna(0).ne(0)) & (chunk["opponentTeamId"].fillna(0).ne(0))]
        if chunk.empty:
            continue
        chunk = chunk.merge(selected_games[["game_id", "season", "season_id", "track", "season_type"]], on="game_id", how="left")
        chunk["game_date"] = pd.to_datetime(chunk["gameDateTimeEst"], errors="coerce")
        chunk = chunk.sort_values(["game_date", "game_id"])
        chunk.to_csv(OUT / "team_game_log.csv", mode="a", header=first, index=False)
        first = False


def write_boxscores(selected_games, team_map):
    chunks = pd.read_csv(PLAYER_STATS_FILE, low_memory=False, chunksize=100000)
    selected_ids = set(selected_games["game_id"])
    first = True
    for chunk in chunks:
        chunk = chunk.rename(columns={"gameId": "game_id"})
        chunk = chunk[chunk["game_id"].isin(selected_ids)]
        if chunk.empty:
            continue
        chunk = chunk.merge(selected_games[["game_id", "season", "season_id", "track", "season_type"]], on="game_id", how="left")
        chunk["game_date"] = pd.to_datetime(chunk["gameDateTimeEst"], errors="coerce")
        chunk = fill_team_ids(
            chunk,
            team_map,
            player_cols=("playerteamCity", "playerteamName"),
            opponent_cols=("opponentteamCity", "opponentteamName"),
        )
        chunk["minutes"] = chunk["numMinutes"].map(parse_minutes)
        chunk["played"] = (chunk["minutes"] > 0).fillna(False).astype(int)
        keep = [
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
        chunk = chunk.reindex(columns=keep)
        chunk = chunk.sort_values(["game_date", "game_id"])
        chunk.to_csv(OUT / "boxscores.csv", mode="a", header=first, index=False)
        first = False


def write_player_statistics_extended(selected_games, team_map):
    chunks = pd.read_csv(PLAYER_STATS_EXTENDED_FILE, low_memory=False, chunksize=100000)
    selected_ids = set(selected_games["game_id"])
    first = True
    for chunk in chunks:
        chunk = chunk.rename(columns={"gameId": "game_id"})
        chunk = chunk[chunk["game_id"].isin(selected_ids)]
        if chunk.empty:
            continue
        chunk = chunk.merge(selected_games[["game_id", "season", "season_id", "track", "season_type"]], on="game_id", how="left")
        chunk["game_date"] = pd.to_datetime(chunk["gameDateTimeEst"], errors="coerce")
        chunk = fill_team_ids(
            chunk,
            team_map,
            player_cols=("playerteamCity", "playerteamName"),
            opponent_cols=("opponentteamCity", "opponentteamName"),
        )
        chunk["minutes"] = chunk["numMinutes"].map(parse_minutes)
        chunk = chunk.sort_values(["game_date", "game_id"])
        chunk.to_csv(OUT / "player_statistics_extended.csv", mode="a", header=first, index=False)
        first = False


def write_team_statistics_extended(selected_games):
    chunks = pd.read_csv(TEAM_STATS_EXTENDED_FILE, low_memory=False, chunksize=100000)
    selected_ids = set(selected_games["game_id"])
    first = True
    for chunk in chunks:
        chunk = chunk.rename(columns={"gameId": "game_id"})
        chunk = chunk[chunk["game_id"].isin(selected_ids)]
        if chunk.empty:
            continue
        chunk = chunk.merge(selected_games[["game_id", "season", "season_id", "track", "season_type"]], on="game_id", how="left")
        chunk["game_date"] = pd.to_datetime(chunk["gameDateTimeEst"], errors="coerce")
        chunk = chunk.sort_values(["game_date", "game_id"])
        chunk.to_csv(OUT / "team_statistics_extended.csv", mode="a", header=first, index=False)
        first = False


def write_players(selected_boxscore_ids):
    data2 = pd.read_csv(PLAYERS_FILE, low_memory=False)
    common = pd.read_csv(COMMON_PLAYER_INFO_FILE, low_memory=False)
    common = common.rename(columns={
        "person_id": "personId",
        "first_name": "firstName_common",
        "last_name": "lastName_common",
        "birthdate": "birthDate_common",
        "school": "school_common",
        "country": "country_common",
        "height": "height_common",
        "weight": "weight_common",
        "position": "position",
        "from_year": "fromYear_common",
        "to_year": "toYear_common",
        "draft_year": "draftYear_common",
        "draft_round": "draftRound_common",
        "draft_number": "draftNumber_common",
    })
    common = common[
        [
            "personId",
            "position",
            "fromYear_common",
            "toYear_common",
            "draftYear_common",
            "draftRound_common",
            "draftNumber_common",
        ]
    ].drop_duplicates(subset=["personId"], keep="first")
    players = data2.merge(common, on="personId", how="left")
    players["position"] = players["position"].where(players["position"].notna(), None)
    players["fromYear"] = players["fromYear"].fillna(players["fromYear_common"])
    players["toYear"] = players["toYear"].fillna(players["toYear_common"])
    players["draftYear"] = players["draftYear"].fillna(players["draftYear_common"])
    players["draftRound"] = players["draftRound"].fillna(players["draftRound_common"])
    players["draftNumber"] = players["draftNumber"].fillna(players["draftNumber_common"])
    players["in_selected_boxscores"] = players["personId"].isin(selected_boxscore_ids).astype(int)
    cols = [
        "personId",
        "firstName",
        "lastName",
        "birthDate",
        "school",
        "country",
        "heightInches",
        "bodyWeightLbs",
        "jersey",
        "position",
        "guard",
        "forward",
        "center",
        "dleagueFlag",
        "nbaFlag",
        "gamesPlayedFlag",
        "draftYear",
        "draftRound",
        "draftNumber",
        "fromYear",
        "toYear",
        "in_selected_boxscores",
    ]
    players = players.reindex(columns=cols)
    players = players.sort_values("personId")
    players.to_csv(OUT / "players.csv", index=False)
    return set(players.loc[players["in_selected_boxscores"] == 1, "personId"])


def write_teams():
    hist = pd.read_csv(TEAM_HISTORIES_FILE, low_memory=False)
    selected_team_ids = set()
    for chunk in pd.read_csv(OUT / "team_game_log.csv", usecols=["teamId", "opponentTeamId"], chunksize=200000):
        selected_team_ids.update(chunk["teamId"].dropna().astype("Int64").tolist())
        selected_team_ids.update(chunk["opponentTeamId"].dropna().astype("Int64").tolist())

    d1 = pd.read_csv(TEAM_HISTORY_FILE, low_memory=False)
    d1 = d1.rename(columns={
        "team_id": "teamId",
        "city": "teamCity",
        "nickname": "teamName",
        "year_founded": "seasonFounded",
        "year_active_till": "seasonActiveTill",
    })
    d1 = d1[d1["teamId"].isin(selected_team_ids)]
    d1 = d1.reindex(columns=["teamId", "teamCity", "teamName", "seasonFounded", "seasonActiveTill"])
    hist = hist[hist["teamId"].isin(selected_team_ids)]
    hist = hist.rename(columns={"teamAbbrev": "abbreviation"})
    teams = pd.concat([hist, d1], ignore_index=True, sort=False)
    teams = teams.drop_duplicates(subset=["teamId", "teamCity", "teamName"], keep="first")
    cols = ["teamId", "teamCity", "teamName", "abbreviation", "seasonFounded", "seasonActiveTill", "league"]
    for col in cols:
        if col not in teams.columns:
            teams[col] = None
    teams = teams.reindex(columns=cols)
    teams = teams.sort_values(["teamId", "seasonFounded"])
    teams.to_csv(OUT / "teams.csv", index=False)


def write_draft_history(player_ids):
    draft = pd.read_csv(DRAFT_HISTORY_FILE, low_memory=False)
    draft = draft[draft["person_id"].isin(player_ids)]
    draft = draft.sort_values(["season", "round_number", "overall_pick", "person_id"])
    draft.to_csv(OUT / "draft_history.csv", index=False)


def write_inactive_players(selected_games):
    inactive = pd.read_csv(INACTIVE_PLAYERS_FILE, low_memory=False)
    inactive = inactive.rename(columns={"game_id": "game_id"})
    inactive = inactive[inactive["game_id"].isin(set(selected_games["game_id"]))]
    inactive = inactive.merge(selected_games[["game_id", "season", "season_id", "track", "season_type"]], on="game_id", how="left")
    inactive = inactive.sort_values(["game_id", "team_id", "player_id"])
    inactive.to_csv(OUT / "inactive_players.csv", index=False)


def write_summary(selected_games):
    summary = (
        selected_games.groupby(["season", "season_id", "track", "season_type"])
        .agg(game_count=("game_id", "nunique"))
        .reset_index()
    )
    summary.to_csv(OUT / "coverage_summary.csv", index=False)


def main():
    for old in OUT.glob("*.csv"):
        old.unlink()


    team_map = build_team_mapping()
    selected_games = read_selected_game_ids()

    print("writing games")
    write_games(selected_games, team_map)
    print("writing team game log")
    write_team_game_log(selected_games)
    print("writing boxscores")
    write_boxscores(selected_games, team_map)
    print("writing player statistics extended")
    write_player_statistics_extended(selected_games, team_map)
    print("writing team statistics extended")
    write_team_statistics_extended(selected_games)
    print("writing players")
    # Read selected player IDs from the newly written boxscores without holding it all in memory.
    box_ids = set()
    for chunk in pd.read_csv(OUT / "boxscores.csv", usecols=["personId"], chunksize=200000):
        box_ids.update(chunk["personId"].dropna().astype("Int64").tolist())
    selected_player_ids = write_players(box_ids)
    print("writing teams")
    write_teams()
    print("writing draft history")
    write_draft_history(selected_player_ids)
    print("writing inactive players")
    write_inactive_players(selected_games)
    print("writing coverage summary")
    write_summary(selected_games)
    print("done")


if __name__ == "__main__":
    main()
