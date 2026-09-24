"""
NBA API Fetcher — reusable, cache-aware wrapper around nba_api.
Phase 1 only uses LeagueGameLog + BoxScoreTraditionalV3.
"""
from __future__ import annotations

import time
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, boxscoretraditionalv3
from nba_api.stats.endpoints import commonallplayers, commonplayerinfo
from nba_api.stats.static import teams


class NBAFetcher:
    """
    Reusable fetcher with per-session in-memory cache.

    Usage:
        f = NBAFetcher()
        games = f.get_season_games('1977-78')
        box = f.get_box_score_traditional('0027700001')
    """

    def __init__(self, delay: float = 0.6):
        self._delay = delay
        self._cache: dict = {}

    def _call(self, cache_key: str, factory):
        if cache_key in self._cache:
            return self._cache[cache_key]
        time.sleep(self._delay)
        result = factory()
        self._cache[cache_key] = result
        return result

    # ---- game list ----

    def get_season_games(
        self, season: str, season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        key = f"leaguegamelog_{season}_{season_type}"
        return self._call(key, lambda: self._fetch_league_game_log(season, season_type))

    def _fetch_league_game_log(self, season: str, season_type: str) -> pd.DataFrame:
        resp = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
        )
        df = resp.get_data_frames()[0]
        print(f"[LeagueGameLog] {season} {season_type}: {len(df)} team-game rows")
        return df

    # ---- box scores ----

    def get_box_score_traditional(self, game_id: str) -> pd.DataFrame:
        key = f"boxscore_trad_{game_id}"
        return self._call(key, lambda: self._fetch_box_traditional(game_id))

    def _fetch_box_traditional(self, game_id: str) -> pd.DataFrame:
        resp = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        df = resp.get_data_frames()[0]
        print(f"[BoxScoreTraditionalV3] {game_id}: {len(df)} player rows")
        return df

    # ---- static ----

    def get_all_teams(self) -> list[dict]:
        return self._call("static_teams", lambda: teams.get_teams())

    def get_common_all_players(self, season: str) -> pd.DataFrame:
        key = f"commonallplayers_{season}"
        return self._call(key, lambda: self._fetch_common_all_players(season))

    def _fetch_common_all_players(self, season: str) -> pd.DataFrame:
        resp = commonallplayers.CommonAllPlayers(season=season, is_only_current_season=1)
        df = resp.get_data_frames()[0]
        print(f"[CommonAllPlayers] {season}: {len(df)} players")
        return df
