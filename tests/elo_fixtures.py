"""Shared synthetic helpers for Elo unit tests."""

import numpy as np
import pandas as pd

GAME_SCORE_COLS = [
    "points",
    "assists",
    "blocks",
    "steals",
    "fieldGoalsAttempted",
    "fieldGoalsMade",
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
]


def make_box_row(player_id, team_id, game_id, home, win, minutes, plus_minus,
                 points=0.0, assists=0.0, blocks=0.0, steals=0.0,
                 fga=0.0, fgm=0.0, fta=0.0, ftm=0.0, oreb=0.0, dreb=0.0,
                 pf=0.0, tov=0.0, game_date="1976-10-21", season=1976,
                 track="regular"):
    """Build one boxscore row with all columns the engine needs."""
    team_minutes = 240.0
    row = {
        "game_id": game_id,
        "game_date": game_date,
        "season": season,
        "track": track,
        "personId": player_id,
        "playerteamId": team_id,
        "home": home,
        "win": win,
        "minutes": minutes,
        "scaled_minutes": minutes,
        "minutes_share": minutes / team_minutes,
        "plusMinusPoints": plus_minus,
        "plus_minus_available": 1,
        "usable_game": 1,
        "played": 1,
        "points": points,
        "assists": assists,
        "blocks": blocks,
        "steals": steals,
        "fieldGoalsAttempted": fga,
        "fieldGoalsMade": fgm,
        "fieldGoalsPercentage": np.nan,
        "threePointersAttempted": 0.0,
        "threePointersMade": 0.0,
        "threePointersPercentage": np.nan,
        "freeThrowsAttempted": fta,
        "freeThrowsMade": ftm,
        "freeThrowsPercentage": np.nan,
        "reboundsDefensive": dreb,
        "reboundsOffensive": oreb,
        "reboundsTotal": oreb + dreb,
        "foulsPersonal": pf,
        "turnovers": tov,
        "startingPosition": "",
        "comment": "",
    }
    return row


def make_game(team_players, game_id=1, game_date="1976-10-21", season=1976,
              home_wins=True, plus_minus_by_player=None, minutes_by_player=None,
              track="regular"):
    """Build a two-team game frame.

    team_players: dict team_id -> list of (player_id, points).
    minutes are split evenly inside each team; plus_minus defaults to 0.
    """
    frames = []
    teams = list(team_players)
    for team_idx, team_id in enumerate(teams):
        players = team_players[team_id]
        is_home = team_idx == 0
        team_min = 240.0
        minutes = minutes_by_player or {pid: team_min / len(players) for pid, _ in players}
        for pid, points in players:
            pm = (plus_minus_by_player or {}).get(pid, 0.0)
            win = 1.0 if (home_wins == is_home) else 0.0
            frames.append(
                make_box_row(
                    pid, team_id, game_id, int(is_home), win,
                    minutes=minutes[pid], plus_minus=pm,
                    points=points, fgm=points / 2.0, fga=points / 2.0 + 2.0,
                    game_date=game_date, season=season, track=track,
                )
            )
    return pd.DataFrame(frames)
