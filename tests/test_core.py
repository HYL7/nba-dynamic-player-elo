"""Unit tests for the pure Elo math functions."""

import math
import unittest

import numpy as np
import pandas as pd

from src.elo.core import (
    apply_zero_sum,
    compute_game_performance,
    elo_delta,
    expected_win_probability,
    game_score,
    k_effective,
    on_court_rate_shrunk,
    perf_i,
    team_rating_average,
    zscore,
    surprise_perf,
)


def _synthetic_game_frame():
    """One game, 8 players (4 per team), with hand-computable +/- totals."""
    rows = [
        # team A: plus-minus sums to +8 over 100 player-minutes
        dict(playerteamId=101, personId=1, minutes=30.0, minutes_share=0.30,
             plusMinusPoints=12.0, points=20.0, fieldGoalsMade=8.0,
             fieldGoalsAttempted=16.0, freeThrowsAttempted=4.0,
             freeThrowsMade=4.0, reboundsOffensive=1.0, reboundsDefensive=5.0,
             assists=4.0, steals=1.0, blocks=1.0, foulsPersonal=2.0,
             turnovers=1.0),
        dict(playerteamId=101, personId=2, minutes=30.0, minutes_share=0.30,
             plusMinusPoints=-6.0, points=10.0, fieldGoalsMade=4.0,
             fieldGoalsAttempted=10.0, freeThrowsAttempted=2.0,
             freeThrowsMade=2.0, reboundsOffensive=0.0, reboundsDefensive=3.0,
             assists=2.0, steals=0.0, blocks=0.0, foulsPersonal=1.0,
             turnovers=2.0),
        dict(playerteamId=101, personId=3, minutes=20.0, minutes_share=0.20,
             plusMinusPoints=4.0, points=8.0, fieldGoalsMade=3.0,
             fieldGoalsAttempted=7.0, freeThrowsAttempted=2.0,
             freeThrowsMade=2.0, reboundsOffensive=1.0, reboundsDefensive=2.0,
             assists=1.0, steals=1.0, blocks=0.0, foulsPersonal=3.0,
             turnovers=0.0),
        dict(playerteamId=101, personId=4, minutes=20.0, minutes_share=0.20,
             plusMinusPoints=-2.0, points=6.0, fieldGoalsMade=2.0,
             fieldGoalsAttempted=6.0, freeThrowsAttempted=2.0,
             freeThrowsMade=2.0, reboundsOffensive=0.0, reboundsDefensive=1.0,
             assists=1.0, steals=0.0, blocks=0.0, foulsPersonal=1.0,
             turnovers=1.0),
        # team B: plus-minus sums to -8 over 100 player-minutes
        dict(playerteamId=102, personId=5, minutes=30.0, minutes_share=0.30,
             plusMinusPoints=-12.0, points=12.0, fieldGoalsMade=5.0,
             fieldGoalsAttempted=12.0, freeThrowsAttempted=3.0,
             freeThrowsMade=3.0, reboundsOffensive=2.0, reboundsDefensive=4.0,
             assists=2.0, steals=1.0, blocks=0.0, foulsPersonal=2.0,
             turnovers=2.0),
        dict(playerteamId=102, personId=6, minutes=30.0, minutes_share=0.30,
             plusMinusPoints=6.0, points=14.0, fieldGoalsMade=6.0,
             fieldGoalsAttempted=12.0, freeThrowsAttempted=1.0,
             freeThrowsMade=1.0, reboundsOffensive=1.0, reboundsDefensive=5.0,
             assists=3.0, steals=0.0, blocks=1.0, foulsPersonal=2.0,
             turnovers=1.0),
        dict(playerteamId=102, personId=7, minutes=20.0, minutes_share=0.20,
             plusMinusPoints=-4.0, points=5.0, fieldGoalsMade=2.0,
             fieldGoalsAttempted=6.0, freeThrowsAttempted=1.0,
             freeThrowsMade=1.0, reboundsOffensive=0.0, reboundsDefensive=2.0,
             assists=1.0, steals=0.0, blocks=0.0, foulsPersonal=1.0,
             turnovers=0.0),
        dict(playerteamId=102, personId=8, minutes=20.0, minutes_share=0.20,
             plusMinusPoints=2.0, points=7.0, fieldGoalsMade=3.0,
             fieldGoalsAttempted=8.0, freeThrowsAttempted=1.0,
             freeThrowsMade=1.0, reboundsOffensive=1.0, reboundsDefensive=2.0,
             assists=1.0, steals=0.0, blocks=0.0, foulsPersonal=1.0,
             turnovers=1.0),
    ]
    return pd.DataFrame(rows)


class GameScoreTest(unittest.TestCase):
    def test_hand_calculated_value(self):
        row = {
            "points": 30.0, "fieldGoalsMade": 10.0, "fieldGoalsAttempted": 20.0,
            "freeThrowsAttempted": 8.0, "freeThrowsMade": 8.0,
            "reboundsOffensive": 2.0, "reboundsDefensive": 6.0,
            "assists": 4.0, "steals": 2.0, "blocks": 1.0,
            "foulsPersonal": 2.0, "turnovers": 3.0,
        }
        expected = (
            30.0 + 0.4 * 10.0 - 0.7 * 20.0 - 0.4 * (8.0 - 8.0)
            + 0.7 * 2.0 + 0.3 * 6.0 + 0.7 * 4.0 + 2.0 + 0.7 * 1.0
            - 0.4 * 2.0 - 3.0
        )
        self.assertAlmostEqual(game_score(row), expected)


class ZScoreTest(unittest.TestCase):
    def test_basic_standardization(self):
        values = [5.0, 6.0, 7.0, 8.0]
        z = zscore(values)
        self.assertAlmostEqual(float(z.mean()), 0.0, places=10)
        self.assertAlmostEqual(float(z.std()), 1.0, places=10)

    def test_sigma_floor(self):
        values = [5.0, 5.0, 5.0, 5.0]
        z = zscore(values)
        self.assertTrue(np.allclose(z, 0.0))


class OnCourtShrinkTest(unittest.TestCase):
    def test_shrink_formula(self):
        # +12 over 36 minutes -> raw 0.3333, shrink factor 36/41.
        value = on_court_rate_shrunk(12.0, 36.0)
        self.assertAlmostEqual(value, (12.0 / 36.0) * (36.0 / 41.0))

    def test_zero_minutes_protected_by_caller(self):
        # NaN/zero handling is done in prepare_game_score_frame; here the
        # raw function just follows the formula and would be +inf for 0.
        value = on_court_rate_shrunk(0.0, 1.0)
        self.assertEqual(value, 0.0)


class OnCourtVariantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = _synthetic_game_frame()
        cls.raw = compute_game_performance(cls.frame, alpha=1.0,
                                           on_court_mode="raw")
        cls.rel = compute_game_performance(cls.frame, alpha=1.0,
                                           on_court_mode="team_relative")
        cls.on_off = compute_game_performance(cls.frame, alpha=1.0,
                                              on_court_mode="on_off")

    def test_raw_is_shrunk_plus_minus_rate(self):
        row = self.frame.iloc[0]
        expected = (row["plusMinusPoints"] / row["minutes"]) * (
            row["minutes"] / (row["minutes"] + 5.0)
        )
        self.assertAlmostEqual(float(self.raw["on_court_rate"].iloc[0]),
                               expected)

    def test_team_relative_compares_to_team_average(self):
        # Team A total +8 over 100 player-minutes -> baseline 0.08.
        p1 = float(self.rel["on_court_rate"].iloc[0])
        expected = (0.4 - 0.08) * (30.0 / 35.0)
        self.assertAlmostEqual(p1, expected, places=10)

    def test_on_off_uses_team_margin_while_sitting(self):
        # p1: on +12/30 = 0.4; off (8-12)/70 = -0.0571429.
        p1 = float(self.on_off["on_court_rate"].iloc[0])
        expected = (0.4 - (-4.0 / 70.0)) * (30.0 / 35.0) * (70.0 / 75.0)
        self.assertAlmostEqual(p1, expected, places=10)

    def test_on_off_full_game_player_is_zero(self):
        # A player who never sits has no off-court sample -> 0.
        frame = self.frame.copy()
        frame.loc[frame["personId"] == 1, "minutes"] = 100.0
        frame.loc[frame["personId"] == 1, "minutes_share"] = 1.0
        teammates = frame["playerteamId"] == 101
        frame.loc[teammates & (frame["personId"] != 1), "minutes"] = 0.0
        frame.loc[teammates & (frame["personId"] != 1), "plusMinusPoints"] = 0.0
        out = compute_game_performance(frame, alpha=1.0, on_court_mode="on_off")
        self.assertAlmostEqual(float(out["on_court_rate"].iloc[0]), 0.0,
                               places=10)

    def test_missing_plus_minus_stays_nan(self):
        frame = self.frame.copy()
        frame.loc[frame["personId"] == 2, "plusMinusPoints"] = np.nan
        for mode in ("raw", "team_relative", "on_off"):
            out = compute_game_performance(frame, alpha=1.0,
                                           on_court_mode=mode)
            self.assertTrue(np.isnan(out["on_court_rate"].iloc[1]))


class ExpectedWinTest(unittest.TestCase):
    def test_symmetric(self):
        self.assertAlmostEqual(
            expected_win_probability(100.0),
            1.0 - expected_win_probability(-100.0),
            places=10,
        )

    def test_equal_ratings_no_home(self):
        self.assertAlmostEqual(expected_win_probability(0.0), 0.5, places=10)

    def test_home_advantage_direction(self):
        self.assertGreater(
            expected_win_probability(0.0, home_advantage=70.0),
            expected_win_probability(0.0),
        )

    def test_bounds(self):
        for diff in [-2000.0, -500.0, -50.0, 0.0, 50.0, 500.0, 2000.0]:
            p = expected_win_probability(diff)
            self.assertGreater(p, 0.0)
            self.assertLess(p, 1.0)


class EloMathTest(unittest.TestCase):
    def test_delta_formula(self):
        # K=20, share=0.5, perf=+2, theta=0.3, S-E=0.2
        d = elo_delta(20.0, 0.5, 2.0, 1.0, 0.8, 0.3)
        self.assertAlmostEqual(d, 20.0 * 0.5 * (2.0 + 0.3 * 0.2))

    def test_zero_sum(self):
        deltas = np.array([5.0, -1.0, 2.0, 0.0])
        adj = apply_zero_sum(deltas)
        self.assertAlmostEqual(float(adj.sum()), 0.0, places=12)
        self.assertAlmostEqual(float(adj.mean()), 0.0, places=12)

    def test_k_rookie_decay(self):
        k1 = k_effective(20.0, 0, rookie_boost=3.0, tau=20.0)
        k2 = k_effective(20.0, 1, rookie_boost=3.0, tau=20.0)
        k20 = k_effective(20.0, 20, rookie_boost=3.0, tau=20.0)
        self.assertAlmostEqual(k1, 60.0, places=10)
        self.assertLess(k2, k1)
        self.assertGreater(k20, 20.0)

    def test_k_rookie_disabled(self):
        self.assertEqual(k_effective(20.0, 0, rookie_boost=1.0), 20.0)

    def test_team_rating_average_weighted(self):
        avg = team_rating_average([1500.0, 1600.0], [0.75, 0.25])
        self.assertAlmostEqual(avg, 1525.0)


class PerfTest(unittest.TestCase):
    def test_alpha_one_ignores_on_court(self):
        gs = [10.0, 20.0, 30.0, 40.0]
        oc = [100.0, -100.0, 50.0, -50.0]
        p1 = perf_i(gs, oc, alpha=1.0)
        p0 = perf_i(gs, [0.0, 0.0, 0.0, 0.0], alpha=1.0)
        self.assertTrue(np.allclose(p1, p0))

    def test_missing_on_court_falls_back_to_game_score(self):
        gs = [10.0, 20.0, 30.0, 40.0]
        p = perf_i(gs, np.array([np.nan] * 4), alpha=0.0)
        self.assertTrue(np.allclose(p, zscore(gs)))

    def test_alpha_zero_is_pure_on_court(self):
        gs = [10.0, 20.0, 30.0, 40.0]
        oc = [5.0, 5.0, 5.0, 5.0]
        p = perf_i(gs, oc, alpha=0.0)
        self.assertTrue(np.allclose(p, 0.0))


class SurprisePerfTest(unittest.TestCase):
    def test_rating_above_anchor_reduces_credit(self):
        # R=1900 with scale=400: perf loses exactly 1.0.
        self.assertAlmostEqual(surprise_perf(1.0, 1900.0, 1500.0, 400.0), 0.0)
        self.assertAlmostEqual(surprise_perf(0.0, 1900.0, 1500.0, 400.0), -1.0)
        self.assertAlmostEqual(surprise_perf(-1.0, 1900.0, 1500.0, 400.0), -2.0)

    def test_rating_below_anchor_boosts_credit(self):
        self.assertAlmostEqual(surprise_perf(-1.0, 1100.0, 1500.0, 400.0), 0.0)
        self.assertAlmostEqual(surprise_perf(0.0, 1100.0, 1500.0, 400.0), 1.0)

    def test_scale_controls_regression_strength(self):
        self.assertAlmostEqual(surprise_perf(0.0, 1500.0, 1500.0, 400.0), 0.0)
        self.assertAlmostEqual(surprise_perf(0.0, 1900.0, 1500.0, 400.0), -1.0)
        self.assertAlmostEqual(surprise_perf(0.0, 1900.0, 1500.0, 800.0), -0.5)


if __name__ == "__main__":
    unittest.main()
