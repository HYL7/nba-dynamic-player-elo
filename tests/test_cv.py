"""Unit tests for the rolling CV metric used in parameter search."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_cv_elo import (
    all_star_cutoff,
    blended_label,
    cv_metric,
    team_win_metrics,
    target_perf,
    top_players_snapshot,
    win_prediction_metrics,
    ratings_as_of_team_game,
    next_window_after_team_games,
)


def boundary_rows(season, pid, rating, perfs, game_offset):
    out = []
    for i, perf in enumerate(perfs, 1):
        out.append({
            "game_id": game_offset + i,
            "game_date": f"{season}-11-{i:02d}",
            "season": season,
            "track": "regular",
            "player_id": pid,
            "rating_after": rating,
            "perf_i": perf,
            "z_game_score": perf,
        })
    return out


class CvMetricTest(unittest.TestCase):
    def test_metric_on_synthetic_boundary(self):
        df = pd.DataFrame(
            boundary_rows(1980, 1, 1600.0, [0.0, 0.0], 100)
            + boundary_rows(1980, 2, 1500.0, [0.0, 0.0], 200)
            + boundary_rows(1981, 1, 1600.0, [0.5, 0.5, 0.5], 300)
            + boundary_rows(1981, 2, 1500.0, [-0.5, -0.5, -0.5], 400)
        )
        metric = cv_metric(
            df, {"scale": 200.0},
            boundary_start=1980, boundary_end=1981,
            window=10, min_games=3,
        )
        # pred = (1600-1550)/200 = 0.25 and (1500-1550)/200 = -0.25;
        # targets are 0.5 and -0.5, so squared errors are 0.0625 each.
        self.assertAlmostEqual(metric["mse"], 0.0625)
        self.assertAlmostEqual(metric["r"], 1.0, places=10)
        # Discrimination check (D-031): pred_sd/target_sd = 0.25/0.5 = 0.5.
        self.assertAlmostEqual(metric["pred_sd"], 0.25)
        self.assertAlmostEqual(metric["target_sd"], 0.5)
        self.assertAlmostEqual(metric["discrimination"], 0.5)
        self.assertEqual(metric["n_players"], 2)
        self.assertEqual(metric["n_boundaries"], 1)

    def test_target_uses_first_window_games(self):
        # The 11th game's perf is extreme and must be excluded by window=10.
        df = pd.DataFrame(
            boundary_rows(1980, 1, 1600.0, [0.0, 0.0], 100)
            + boundary_rows(1980, 2, 1500.0, [0.0, 0.0], 200)
            + boundary_rows(1981, 1, 1600.0, [0.5] * 10, 300)
            + boundary_rows(1981, 2, 1500.0, [-0.5] * 10, 400)
            + boundary_rows(1981, 1, 1600.0, [99.0], 500)
        )
        # Move the extreme game after the first ten by date order.
        df.loc[df["game_id"] == 501, "game_date"] = "1981-12-01"
        metric = cv_metric(
            df, {"scale": 200.0},
            boundary_start=1980, boundary_end=1981,
            window=10, min_games=3,
        )
        self.assertAlmostEqual(metric["mse"], 0.0625)
        self.assertEqual(metric["n_players"], 2)

    def test_target_is_fixed_alpha_ref_label(self):
        # D-027a: alpha_modern must not move the CV target. The label is the
        # stored pure GameScore z even when engine perf_i blends in on-court.
        df = pd.DataFrame(
            boundary_rows(1980, 1, 1600.0, [0.0, 0.0], 100)
            + boundary_rows(1980, 2, 1500.0, [0.0, 0.0], 200)
            + boundary_rows(1981, 1, 1600.0, [0.5] * 10, 300)
            + boundary_rows(1981, 2, 1500.0, [-0.5] * 10, 400)
        )
        # Simulate an alpha=0 engine: blended perf_i is compressed toward 0.
        df.loc[df["season"] == 1981, "perf_i"] *= 0.2
        metric = cv_metric(
            df, {"scale": 200.0},
            boundary_start=1980, boundary_end=1981,
            window=10, min_games=3,
        )
        self.assertAlmostEqual(metric["target_sd"], 0.5)
        self.assertAlmostEqual(metric["mse"], 0.0625)

    def test_win_prediction_metrics_log_loss_and_brier(self):
        df = pd.DataFrame([
            {"game_id": 1, "game_date": "1980-11-01", "player_id": 1,
             "team_id": 10, "s_team": 1, "e_team": 0.75},
            {"game_id": 1, "game_date": "1980-11-01", "player_id": 2,
             "team_id": 20, "s_team": 0, "e_team": 0.25},
            {"game_id": 2, "game_date": "1980-11-02", "player_id": 1,
             "team_id": 10, "s_team": 0, "e_team": 0.40},
            {"game_id": 2, "game_date": "1980-11-02", "player_id": 2,
             "team_id": 20, "s_team": 1, "e_team": 0.60},
        ])
        metric = win_prediction_metrics(df)
        self.assertAlmostEqual(metric["win_log_loss"], 0.399254, places=6)
        self.assertAlmostEqual(metric["win_brier"], 0.11125, places=6)
        self.assertEqual(metric["win_games"], 2)


    def test_alpha_ref_blended_label(self):
        df = pd.DataFrame([
            {"game_id": 1, "game_date": "1981-11-01", "season": 1981,
             "track": "regular", "player_id": 1, "rating_after": 1600.0,
             "z_game_score": 0.4, "z_on_court_rate": 2.0},
            {"game_id": 1, "game_date": "1981-11-01", "season": 1981,
             "track": "regular", "player_id": 2, "rating_after": 1500.0,
             "z_game_score": -0.4, "z_on_court_rate": -2.0},
        ])
        t1 = target_perf(df, window=10, min_games=1, alpha_ref=1.0)
        t0 = target_perf(df, window=10, min_games=1, alpha_ref=0.0)
        t05 = target_perf(df, window=10, min_games=1, alpha_ref=0.5)
        by_id = {int(r.player_id): r.target for r in t1.itertuples()}
        self.assertAlmostEqual(by_id[1], 0.4)
        self.assertAlmostEqual(by_id[2], -0.4)
        by_id0 = {int(r.player_id): r.target for r in t0.itertuples()}
        self.assertAlmostEqual(by_id0[1], 2.0)
        self.assertAlmostEqual(by_id0[2], -2.0)
        by_id05 = {int(r.player_id): r.target for r in t05.itertuples()}
        self.assertAlmostEqual(by_id05[1], 1.2)
        self.assertAlmostEqual(by_id05[2], -1.2)

    def test_alpha_ref_missing_on_court_falls_back_to_game_score(self):
        df = pd.DataFrame([
            {"game_id": 1, "game_date": "1981-11-01", "season": 1981,
             "track": "regular", "player_id": 1, "rating_after": 1600.0,
             "z_game_score": 0.4, "z_on_court_rate": np.nan},
            {"game_id": 1, "game_date": "1981-11-01", "season": 1981,
             "track": "regular", "player_id": 2, "rating_after": 1500.0,
             "z_game_score": -0.4, "z_on_court_rate": np.nan},
        ])
        t0 = target_perf(df, window=10, min_games=1, alpha_ref=0.0)
        by_id = {int(r.player_id): r.target for r in t0.itertuples()}
        self.assertAlmostEqual(by_id[1], 0.4)
        self.assertAlmostEqual(by_id[2], -0.4)

    def test_team_win_metrics_correlation(self):
        rows = []
        outcomes = [(1, 0), (1, 0), (1, 0), (0, 1)]
        for i, (s10, s20) in enumerate(outcomes, 1):
            for pid in (1, 2):
                rows.append({"game_id": 100 + i, "game_date": f"2000-11-0{i}",
                             "season": 2000, "track": "regular", "player_id": pid,
                             "team_id": 10, "s_team": s10, "e_team": 0.7})
            for pid in (3, 4):
                rows.append({"game_id": 100 + i, "game_date": f"2000-11-0{i}",
                             "season": 2000, "track": "regular", "player_id": pid,
                             "team_id": 20, "s_team": s20, "e_team": 0.3})
        metric = team_win_metrics(pd.DataFrame(rows))
        self.assertAlmostEqual(metric["team_win_corr"], 1.0, places=10)
        self.assertEqual(metric["team_win_pairs"], 2)
        self.assertEqual(metric["team_win_games"], 8)

    def test_all_star_cutoff_dates(self):
        self.assertEqual(all_star_cutoff(1997), "1998-02-01")
        self.assertEqual(all_star_cutoff(1998), "1999-02-14")
        self.assertEqual(all_star_cutoff(2020), "2021-03-01")

    def test_top_players_snapshot(self):
        df = pd.DataFrame(
            boundary_rows(1997, 1, 1600.0, [0.0, 0.0], 100)
            + boundary_rows(1997, 2, 1500.0, [0.0, 0.0], 200)
            + boundary_rows(1997, 3, 1400.0, [0.0, 0.0], 300)
        )
        top = top_players_snapshot(df, seasons=(1997,), top_n=2)
        self.assertEqual(list(top["player_id"]), [1, 2])
        self.assertEqual(list(top["rank"]), [1, 2])
        self.assertEqual(list(top["season_end"]), [1998, 1998])

    def test_ratings_as_of_team_game_cutoff(self):
        rows = []
        for g in range(1, 4):
            for pid in (1, 2):
                rows.append({"game_id": 100 + g, "game_date": f"2000-11-0{g}",
                             "season": 2000, "track": "regular", "player_id": pid,
                             "team_id": 10, "rating_after": 1500.0 + g * 10})
            for pid in (3, 4):
                rows.append({"game_id": 100 + g, "game_date": f"2000-11-0{g}",
                             "season": 2000, "track": "regular", "player_id": pid,
                             "team_id": 20, "rating_after": 1400.0 + g * 10})
        # Playoff updates are ignored by the regular-season cutoff.
        rows.append({"game_id": 999, "game_date": "2001-05-01", "season": 2000,
                     "track": "playoffs", "player_id": 1, "team_id": 10,
                     "rating_after": 9999.0})
        df = pd.DataFrame(rows)
        out = ratings_as_of_team_game(df, 2)
        by = {(int(r.season), int(r.player_id)): r.rating for r in out.itertuples()}
        self.assertEqual(by[(2000, 1)], 1520.0)
        self.assertEqual(by[(2000, 3)], 1420.0)
        self.assertEqual(len(by), 4)
        self.assertAlmostEqual(
            out.loc[out["player_id"] == 1, "league_mean"].iloc[0], 1470.0)

    def test_ratings_as_of_team_game_short_season_fallback(self):
        rows = [
            {"game_id": 1, "game_date": "1999-02-05", "season": 1998,
             "track": "regular", "player_id": 1, "team_id": 10,
             "rating_after": 1600.0},
            {"game_id": 1, "game_date": "1999-02-05", "season": 1998,
             "track": "regular", "player_id": 2, "team_id": 20,
             "rating_after": 1400.0},
        ]
        out = ratings_as_of_team_game(pd.DataFrame(rows), 60)
        by = {(int(r.player_id)): r.rating for r in out.itertuples()}
        self.assertEqual(by[1], 1600.0)
        self.assertEqual(by[2], 1400.0)

    def test_next_window_after_team_games(self):
        # Player 3 appears only in the first two games and must be dropped
        # because there is no next window after the game-2 cutoff.
        rows = []
        for g in range(1, 5):
            rows.append({"game_id": 100 + g, "game_date": f"2000-11-0{g}",
                         "season": 2000, "track": "regular", "player_id": 1,
                         "team_id": 10, "z_game_score": g * 0.1})
            if g <= 2:
                rows.append({"game_id": 100 + g, "game_date": f"2000-11-0{g}",
                             "season": 2000, "track": "regular", "player_id": 3,
                             "team_id": 10, "z_game_score": 1.0})
            rows.append({"game_id": 100 + g, "game_date": f"2000-11-0{g}",
                         "season": 2000, "track": "regular", "player_id": 2,
                         "team_id": 20, "z_game_score": -g * 0.1})
        out = next_window_after_team_games(
            pd.DataFrame(rows), 2, window=10, min_games=2)
        by = {int(r.player_id): r.target for r in out.itertuples()}
        counts = {int(r.player_id): int(r.count) for r in out.itertuples()}
        self.assertAlmostEqual(by[1], 0.35)
        self.assertAlmostEqual(by[2], -0.35)
        self.assertNotIn(3, by)
        self.assertEqual(counts[1], 2)

    def test_cv_metric_mid_season_uses_same_season_target(self):
        rows = []
        for g in range(1, 4):
            rows.append({"game_id": 100 + g, "game_date": f"2000-11-0{g}",
                         "season": 2000, "track": "regular", "player_id": 1,
                         "team_id": 10, "rating_after": 1600.0,
                         "z_game_score": 0.5})
            rows.append({"game_id": 100 + g, "game_date": f"2000-11-0{g}",
                         "season": 2000, "track": "regular", "player_id": 2,
                         "team_id": 20, "rating_after": 1500.0,
                         "z_game_score": -0.5})
        metric = cv_metric(
            pd.DataFrame(rows), {"scale": 200.0, "alpha_ref": 1.0},
            boundary_start=2000, boundary_end=2000, window=10, min_games=1,
            sampling="game2")
        self.assertAlmostEqual(metric["mse"], 0.0625)
        self.assertAlmostEqual(metric["r"], 1.0, places=10)
        self.assertEqual(metric["n_players"], 2)
        self.assertEqual(metric["n_boundaries"], 1)


if __name__ == "__main__":
    unittest.main()
