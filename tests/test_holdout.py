"""Unit tests for the Step 5 hold-out pipeline (scripts/run_holdout.py)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_holdout import (
    build_elo_pairs,
    random_pairs,
    summarize,
    team_elo_ratings,
)


def synthetic_updates():
    """Two players across a boundary season and a target season."""
    return pd.DataFrame([
        # Boundary season 2022: end ratings define pred.
        {"game_id": 100, "game_date": "2022-11-01", "season": 2022,
         "track": "regular", "player_id": 1, "team_id": 10,
         "home": 0, "s_team": 0, "rating_after": 1700.0,
         "z_game_score": 0.2, "z_on_court_rate": 0.0},
        {"game_id": 100, "game_date": "2022-11-01", "season": 2022,
         "track": "regular", "player_id": 2, "team_id": 20,
         "home": 1, "s_team": 1, "rating_after": 1500.0,
         "z_game_score": -0.2, "z_on_court_rate": 0.0},
        # Target season 2023: first ten regular games are the target window.
        {"game_id": 200, "game_date": "2023-11-01", "season": 2023,
         "track": "regular", "player_id": 1, "team_id": 10,
         "home": 0, "s_team": 0, "rating_after": 1700.0,
         "z_game_score": 0.5, "z_on_court_rate": 0.0},
        {"game_id": 200, "game_date": "2023-11-01", "season": 2023,
         "track": "regular", "player_id": 2, "team_id": 20,
         "home": 1, "s_team": 1, "rating_after": 1500.0,
         "z_game_score": -0.5, "z_on_court_rate": 0.0},
    ])


class BuildEloPairsTest(unittest.TestCase):
    def test_pred_formula_and_target_window(self):
        df = synthetic_updates()
        pairs = build_elo_pairs(df, scale=200.0, alpha_ref=1.0,
                                boundary_start=2022, boundary_end=2022,
                                window=10, min_games=1)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(sorted(pairs["boundary_season"]), [2022, 2022])
        # league mean is (1700+1500)/2 = 1600, so preds are 0.5 and -0.5.
        preds = dict(zip(pairs["player_id"], pairs["pred"]))
        self.assertAlmostEqual(preds[1], 0.5)
        self.assertAlmostEqual(preds[2], -0.5)
        targets = dict(zip(pairs["player_id"], pairs["target"]))
        self.assertAlmostEqual(targets[1], 0.5)
        self.assertAlmostEqual(targets[2], -0.5)

    def test_alpha_ref_label_is_pure_game_score(self):
        df = synthetic_updates()
        # Even when z_on_court_rate disagrees, alpha_ref=1.0 keeps GS label.
        df.loc[df["season"] == 2023, "z_on_court_rate"] = 5.0
        pairs = build_elo_pairs(df, scale=200.0, alpha_ref=1.0,
                                boundary_start=2022, boundary_end=2022,
                                window=10, min_games=1)
        targets = dict(zip(pairs["player_id"], pairs["target"]))
        self.assertAlmostEqual(targets[1], 0.5)
        self.assertAlmostEqual(targets[2], -0.5)


class RandomPairsTest(unittest.TestCase):
    def test_permutation_preserves_targets(self):
        df = synthetic_updates()
        base = build_elo_pairs(df, scale=200.0, alpha_ref=1.0,
                               boundary_start=2022, boundary_end=2022,
                               window=10, min_games=1)
        rnd = random_pairs(base, seed=12345)
        self.assertEqual(
            sorted(rnd["target"].round(6)),
            sorted(base["target"].round(6)),
        )
        self.assertEqual(len(rnd), len(base))


class TeamEloTest(unittest.TestCase):
    def _team_frame(self):
        # Two games, one win per team; each game has two duplicate player rows.
        return pd.DataFrame([
            {"game_id": 1, "game_date": "2022-11-01", "season": 2022,
             "track": "regular", "team_id": 10, "home": 1, "s_team": 1},
            {"game_id": 1, "game_date": "2022-11-01", "season": 2022,
             "track": "regular", "team_id": 20, "home": 0, "s_team": 0},
            {"game_id": 2, "game_date": "2022-11-03", "season": 2022,
             "track": "regular", "team_id": 10, "home": 0, "s_team": 1},
            {"game_id": 2, "game_date": "2022-11-03", "season": 2022,
             "track": "regular", "team_id": 20, "home": 1, "s_team": 0},
        ])

    def test_team_elo_zero_sum_and_direction(self):
        df = self._team_frame()
        end = team_elo_ratings(df, k=20.0, home_advantage=70.0)
        ratings = end[end["season"] == 2022].set_index("team_id")["team_rating"]
        # Ratings sum to 2 * 1500 because each game is zero-sum.
        self.assertAlmostEqual(ratings.sum(), 3000.0, places=9)
        self.assertAlmostEqual(float(end["team_league_mean"].iloc[0]), 1500.0)
        # Team 10 wins twice (home then away), so it finishes above team 20.
        self.assertGreater(ratings[10], ratings[20])

    def test_build_team_pairs(self):
        from run_holdout import build_team_pairs
        df = synthetic_updates()
        # Add team identity to the target-season rows for the player-team map.
        pairs = build_team_pairs(df, scale=100.0, alpha_ref=1.0,
                                 boundary_start=2022, boundary_end=2022,
                                 window=10, min_games=1, team_k=20.0)
        self.assertEqual(len(pairs), 2)
        # Winning team 20 gets the higher rating, so its player has higher pred.
        preds = dict(zip(pairs["player_id"], pairs["pred"]))
        self.assertGreater(preds[2], preds[1])


class SummarizeTest(unittest.TestCase):
    def test_perfect_prediction(self):
        pairs = pd.DataFrame({
            "boundary_season": [2022, 2022, 2022, 2023, 2023, 2023],
            "player_id": [1, 2, 3, 1, 2, 4],
            "pred": [0.6, -0.1, -0.5, 0.4, 0.0, -0.4],
            "target": [0.6, -0.1, -0.5, 0.4, 0.0, -0.4],
        })
        stat = summarize(pairs, n_iter=100, seed=1)
        self.assertAlmostEqual(stat["mse"], 0.0)
        self.assertAlmostEqual(stat["r"], 1.0)
        self.assertAlmostEqual(stat["improvement"], stat["baseline_mse"])
        self.assertEqual(stat["n_positive_r"], 2)
        self.assertEqual(stat["n_boundaries"], 2)
        for key in [
            "boot_player_r_mean", "boot_player_gain_mean",
            "boot_season_r_mean", "boot_season_gain_mean",
        ]:
            self.assertIn(key, stat)


if __name__ == "__main__":
    unittest.main()
