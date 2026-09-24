"""Sequential Dynamic Player Elo engine.

Consumes a clean boxscore frame (only usable_game == 1 rows) ordered by
game_date / game_id and produces per-game update records plus player state.
The engine is intentionally independent of CSV loading so it can be tested
on synthetic frames and run on clean_data in the same way.
"""

import numpy as np
import pandas as pd

from .core import (
    compute_game_performance,
    elo_delta,
    expected_win_probability,
    k_effective,
    apply_zero_sum,
    surprise_perf,
    team_rating_average,
)

UPDATE_COLUMNS = [
    "game_id",
    "game_date",
    "season",
    "track",
    "player_id",
    "team_id",
    "home",
    "rating_before",
    "rating_after",
    "delta_raw",
    "delta_adj",
    "zero_mean_offset",
    "minutes_share",
    "perf_i",
    "perf_used",
    "game_score",
    "z_game_score",
    "on_court_rate",
    "z_on_court_rate",
    "s_team",
    "e_team",
    "k_effective",
]

STATE_COLUMNS = [
    "game_id",
    "game_date",
    "track",
    "player_id",
    "rating",
    "games_played",
    "games_played_track",
]


def build_draft_rating_map(players, pick_col="overall_pick", **params):
    """Map personId -> initial rating from draft pick (linear decline)."""
    if pick_col not in players.columns:
        raise ValueError(f"draft map source needs column {pick_col}")
    rating_map = {}
    for pid, pick in zip(players["person_id"], players[pick_col]):
        if pd.isna(pick):
            rating_map[int(pid)] = params.get("rookie_floor", 1350.0)
            continue
        rating = params.get("pick1_rating", 1580.0) - (float(pick) - 1.0) * params.get("pick_slope", 5.0)
        rating = max(rating, params.get("rookie_floor", 1350.0))
        rating_map[int(pid)] = rating
    return rating_map


class EloEngine:
    def __init__(self, k=20.0, theta=0.3, home_advantage=70.0, alpha=1.0,
                 rookie_boost=3.0, rookie_tau=25.0, init_mode="uniform",
                 init_value=1500.0, draft_ratings=None, sigma_floor=1.0,
                 rookie_ids=None, record_states=True,
                 surprise_scale=None, surprise_anchor="game",
                 playoff_k_multiplier=1.0,
                 alpha_modern=None, alpha_modern_start_season=1996,
                 on_court_mode="raw",
                 initial_state=None,
                 reverse=False):
        self.k = float(k)
        self.theta = float(theta)
        self.home_advantage = float(home_advantage)
        self.alpha = float(alpha)
        self.rookie_boost = float(rookie_boost)
        self.rookie_tau = float(rookie_tau)
        self.sigma_floor = float(sigma_floor)
        if on_court_mode not in ("raw", "team_relative", "on_off"):
            raise ValueError(f"unknown on_court_mode: {on_court_mode}")
        self.on_court_mode = on_court_mode
        self.init_mode = init_mode
        self.init_value = float(init_value)
        self.draft_ratings = dict(draft_ratings or {})
        self.rookie_ids = set(rookie_ids or []) if rookie_ids is not None else None
        self.record_states = bool(record_states)
        self.surprise_scale = float(surprise_scale) if surprise_scale is not None else None
        if self.surprise_scale is not None and self.surprise_scale <= 0:
            raise ValueError("surprise_scale must be positive")
        self.surprise_anchor = surprise_anchor
        self.playoff_k_multiplier = float(playoff_k_multiplier)
        if self.playoff_k_multiplier < 0:
            raise ValueError("playoff_k_multiplier must be non-negative")
        self.alpha_modern = float(alpha_modern) if alpha_modern is not None else None
        if self.alpha_modern is not None and not 0.0 <= self.alpha_modern <= 1.0:
            raise ValueError("alpha_modern must be in [0, 1]")
        self.alpha_modern_start_season = int(alpha_modern_start_season)
        self.initial_state = dict(initial_state or {})
        self.reverse = bool(reverse)

    def _init_rating(self, player_id):
        if self.init_mode == "uniform":
            return self.init_value, "uniform"
        if self.init_mode == "draft":
            rating = self.draft_ratings.get(int(player_id))
            if rating is None:
                raise ValueError(f"no draft rating for player {player_id}")
            return float(rating), "draft"
        raise ValueError(f"unknown init_mode: {self.init_mode}")

    def _surprise_anchor_value(self, ratings, games_played, game):
        if self.surprise_scale is None:
            return None
        if self.surprise_anchor == "fixed":
            return 1500.0
        if self.surprise_anchor == "game":
            return float(np.mean([ratings[int(p)] for p in game["personId"]]))
        if self.surprise_anchor == "league":
            active = [r for pid, r in ratings.items() if games_played[pid] > 0]
            return float(np.mean(active)) if active else 1500.0
        raise ValueError(f"unknown surprise_anchor: {self.surprise_anchor}")

    def run(self, boxscores, players=None):
        """Run the Elo state machine over a clean boxscore frame.

        Parameters
        ----------
        boxscores : DataFrame with at least the update columns
            (game_id, game_date, track, personId, playerteamId, home, win,
            minutes_share, plusMinusPoints, plus_minus_available and all
            GameScore stat columns). Only usable_game == 1 rows should be
            passed.
        players : optional DataFrame with personId and draft info, used only
            for init_mode='draft' when a player is not in draft_ratings.

        surprise_scale, when set, replaces perf_i with
        perf_i - (rating - anchor) / surprise_scale. Anchor is one of
        "fixed" (1500), "game" (mean rating of the current game; the
        default, matching the narrative of performance relative to the
        game's expected level) or "league" (mean rating of all tracked
        players).

        playoff_k_multiplier scales K for rows whose track is "playoffs".
        Default 1.0 keeps regular and playoff updates at equal weight; 0.0
        keeps playoff games in the timeline but gives them no Elo impact.
        The multiplier is the last tuning knob for postseason impact (D-025).

        alpha_modern, when set, switches the GameScore/on-court blend to
        alpha_modern for seasons >= alpha_modern_start_season (1996-97 is
        the first season with real on-court +/-). Seasons before that keep
        self.alpha, so the pipeline can run one coherent engine across the
        two data-quality eras.

        reverse=True processes games newest-to-oldest with the same formula,
        used by the D-007 backward pass to derive 1976-77 start ratings.

        Returns
        -------
        (updates, states, inits) : DataFrames
        """
        df = boxscores.copy()
        required = {"game_id", "game_date", "track", "personId", "playerteamId",
                    "home", "win", "minutes_share", "plusMinusPoints",
                    "plus_minus_available"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        df = df.sort_values(
            ["game_date", "game_id", "personId"],
            ascending=[not self.reverse, not self.reverse, True],
        ).reset_index(drop=True)

        ratings = {}
        games_played = {}
        games_played_track = {}
        inits = []
        updates = []
        states = []

        for pid, (rating, played, played_track) in self.initial_state.items():
            ratings[int(pid)] = float(rating)
            games_played[int(pid)] = int(played)
            games_played_track[int(pid)] = int(played_track)

        for game_id, game in df.groupby("game_id", sort=False):
            game = game.sort_values("personId")
            game_id = int(game_id)
            game_date = game["game_date"].iloc[0]
            track = game["track"].iloc[0]
            season = game["season"].iloc[0] if "season" in game.columns else None

            # Initialize players making their debut in this game.
            for pid in game["personId"]:
                pid = int(pid)
                if pid not in ratings:
                    rating, source = self._init_rating(pid)
                    ratings[pid] = rating
                    games_played[pid] = 0
                    games_played_track[pid] = 0
                    inits.append({
                        "player_id": pid,
                        "game_id": game_id,
                        "game_date": game_date,
                        "rating": rating,
                        "source": source,
                    })

            alpha_eff = self.alpha
            if (
                self.alpha_modern is not None
                and season is not None
                and int(season) >= self.alpha_modern_start_season
            ):
                alpha_eff = self.alpha_modern
            perf = compute_game_performance(
                game, alpha=alpha_eff, sigma_floor=self.sigma_floor,
                on_court_mode=self.on_court_mode,
            )

            # Team averages weighted by minutes share.
            team_avgs = {}
            for team_id, team_frame in perf.groupby("playerteamId"):
                team_id = int(team_id)
                r = np.array([ratings[int(p)] for p in team_frame["personId"]])
                m = team_frame["minutes_share"].to_numpy(dtype=float)
                team_avgs[team_id] = team_rating_average(r, m)
            if len(team_avgs) != 2:
                raise ValueError(f"game {game_id} does not have exactly 2 teams: {sorted(team_avgs)}")

            # Expected win probability from the perspective of each team.
            e_by_team = {}
            for team_id in team_avgs:
                opp_id = [t for t in team_avgs if t != team_id][0]
                rating_diff = team_avgs[team_id] - team_avgs[opp_id]
                advantage = self.home_advantage if int(perf.loc[perf["playerteamId"] == team_id, "home"].iloc[0]) == 1 else 0.0
                e_by_team[team_id] = expected_win_probability(
                    rating_diff, home_advantage=advantage
                )

            surprise_anchor = self._surprise_anchor_value(ratings, games_played, game)

            raw_deltas = []
            for row in perf.itertuples(index=False):
                team_id = int(row.playerteamId)
                s_team = float(row.win)
                e_team = e_by_team[team_id]
                pid = int(row.personId)
                boost = self.rookie_boost if self.rookie_ids is None or pid in self.rookie_ids else 1.0
                k = k_effective(
                    self.k,
                    games_played[pid],
                    rookie_boost=boost,
                    tau=self.rookie_tau,
                )
                if track == "playoffs":
                    k *= self.playoff_k_multiplier
                perf_value = float(row.perf_i)
                if surprise_anchor is not None:
                    perf_value = surprise_perf(
                        perf_value, ratings[pid], surprise_anchor, self.surprise_scale
                    )
                raw_deltas.append(
                    elo_delta(k, float(row.minutes_share), perf_value,
                              s_team, e_team, self.theta)
                )
            adj_deltas = apply_zero_sum(raw_deltas)
            mean_offset = float(np.mean(raw_deltas))

            for (row, raw, adj) in zip(perf.itertuples(index=False), raw_deltas, adj_deltas):
                pid = int(row.personId)
                team_id = int(row.playerteamId)
                s_team = float(row.win)
                e_team = e_by_team[team_id]
                boost = self.rookie_boost if self.rookie_ids is None or pid in self.rookie_ids else 1.0
                k = k_effective(
                    self.k,
                    games_played[pid],
                    rookie_boost=boost,
                    tau=self.rookie_tau,
                )
                if track == "playoffs":
                    k *= self.playoff_k_multiplier
                perf_value = float(row.perf_i)
                if surprise_anchor is not None:
                    perf_value = surprise_perf(
                        perf_value, ratings[pid], surprise_anchor, self.surprise_scale
                    )
                before = ratings[pid]
                after = before + float(adj)
                ratings[pid] = after
                games_played[pid] += 1
                games_played_track[pid] += 1
                updates.append({
                    "game_id": game_id,
                    "game_date": game_date,
                    "season": season,
                    "track": track,
                    "player_id": pid,
                    "team_id": team_id,
                    "home": int(row.home),
                    "rating_before": before,
                    "rating_after": after,
                    "delta_raw": float(raw),
                    "delta_adj": float(adj),
                    "zero_mean_offset": mean_offset,
                    "minutes_share": float(row.minutes_share),
                    "perf_i": float(row.perf_i),
                    "perf_used": perf_value,
                    "game_score": float(row.game_score),
                    "z_game_score": float(row.z_game_score),
                    "on_court_rate": float(row.on_court_rate) if pd.notna(row.on_court_rate) else None,
                    "z_on_court_rate": float(row.z_on_court_rate) if pd.notna(row.z_on_court_rate) else None,
                    "s_team": s_team,
                    "e_team": e_team,
                    "k_effective": k,
                })

            if self.record_states:
                for pid, rating in ratings.items():
                    if games_played[pid] == 0:
                        continue
                    states.append({
                        "game_id": game_id,
                        "game_date": game_date,
                        "track": track,
                        "player_id": pid,
                        "rating": rating,
                        "games_played": games_played[pid],
                        "games_played_track": games_played_track[pid],
                    })

        return (
            pd.DataFrame(updates, columns=UPDATE_COLUMNS),
            pd.DataFrame(states, columns=STATE_COLUMNS),
            pd.DataFrame(inits),
        )
