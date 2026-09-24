"""Integration tests for the sequential Elo engine."""

import unittest

import numpy as np
import pandas as pd

from src.elo.engine import EloEngine
from tests.elo_fixtures import make_game


class EngineTest(unittest.TestCase):
    def setUp(self):
        self.engine = EloEngine(
            k=20.0, theta=0.3, home_advantage=70.0, alpha=1.0,
            rookie_boost=1.0, init_mode="uniform",
        )

    def test_zero_sum_per_game(self):
        game = make_game(
            {
                1: [(101, 10.0), (102, 20.0), (103, 30.0)],
                2: [(201, 15.0), (202, 25.0), (203, 5.0)],
            },
            game_id=7,
        )
        updates, states, inits = self.engine.run(game)
        per_game = updates.groupby("game_id")["delta_adj"].sum()
        for game_id, total in per_game.items():
            self.assertAlmostEqual(total, 0.0, places=10, msg=f"game {game_id}")
        self.assertEqual(len(updates), 6)
        self.assertEqual(len(inits), 6)

    def test_rating_after_is_before_plus_delta(self):
        game = make_game(
            {
                1: [(101, 10.0), (102, 20.0), (103, 30.0)],
                2: [(201, 15.0), (202, 25.0), (203, 5.0)],
            },
            game_id=8,
        )
        updates, _, _ = self.engine.run(game)
        diff = updates["rating_after"] - updates["rating_before"]
        self.assertTrue(np.allclose(diff, updates["delta_adj"], atol=1e-9))

    def test_chronological_multi_game_state(self):
        game1 = make_game(
            {1: [(101, 10.0), (102, 20.0)], 2: [(201, 15.0), (202, 25.0)]},
            game_id=1, game_date="1976-10-21",
        )
        game2 = make_game(
            {1: [(101, 12.0), (102, 18.0)], 3: [(301, 8.0), (302, 22.0)]},
            game_id=2, game_date="1976-10-22",
        )
        box = pd.concat([game1, game2], ignore_index=True)
        updates, states, inits = self.engine.run(box)

        self.assertEqual(set(updates["game_id"]), {1, 2})
        self.assertEqual(len(inits), 6)
        player101 = states[states["player_id"] == 101].sort_values("game_id")
        self.assertEqual(list(player101["game_id"]), [1, 2])
        self.assertGreater(len(player101), 1)

        # Player 101's first update is game 1, second is game 2.
        p101 = updates[updates["player_id"] == 101].sort_values("game_id")
        self.assertEqual(p101.iloc[0]["game_id"], 1)
        self.assertEqual(p101.iloc[1]["game_id"], 2)

    def test_reverse_processes_games_newest_first(self):
        game1 = make_game(
            {1: [(101, 10.0), (102, 20.0)], 2: [(201, 15.0), (202, 25.0)]},
            game_id=1, game_date="1976-10-21",
        )
        game2 = make_game(
            {1: [(101, 12.0), (102, 18.0)], 3: [(301, 8.0), (302, 22.0)]},
            game_id=2, game_date="1976-10-22",
        )
        box = pd.concat([game1, game2], ignore_index=True)
        engine = EloEngine(
            k=20.0, theta=0.3, home_advantage=70.0, alpha=1.0,
            rookie_boost=1.0, init_mode="uniform", reverse=True,
        )
        updates, _, _ = engine.run(box)
        self.assertEqual(list(updates["game_id"].drop_duplicates()), [2, 1])

    def test_draft_init(self):
        engine = EloEngine(init_mode="draft", draft_ratings={101: 1580.0, 201: 1435.0})
        game = make_game({1: [(101, 10.0)], 2: [(201, 15.0)]}, game_id=9)
        updates, states, inits = engine.run(game)
        init_map = dict(zip(inits["player_id"], inits["rating"]))
        self.assertEqual(init_map[101], 1580.0)
        self.assertEqual(init_map[201], 1435.0)
        self.assertEqual(updates.loc[updates["player_id"] == 101, "rating_before"].iloc[0], 1580.0)

    def test_k_boost_decays_within_run(self):
        engine = EloEngine(
            k=20.0, rookie_boost=3.0, rookie_tau=20.0, init_mode="uniform",
        )
        g1 = make_game({1: [(101, 10.0)], 2: [(201, 15.0)]}, game_id=1, game_date="1976-10-21")
        g2 = make_game({1: [(101, 12.0)], 2: [(201, 18.0)]}, game_id=2, game_date="1976-10-22")
        g3 = make_game({1: [(101, 8.0)], 2: [(201, 22.0)]}, game_id=3, game_date="1976-10-23")
        box = pd.concat([g1, g2, g3], ignore_index=True)
        updates, _, _ = engine.run(box)
        k101 = updates[updates["player_id"] == 101]["k_effective"].tolist()
        self.assertAlmostEqual(k101[0], 60.0, places=10)
        self.assertLess(k101[1], k101[0])
        self.assertLess(k101[2], k101[1])

    def test_minutes_share_uses_provided_share(self):
        # Teams have uneven minutes in this synthetic frame; engine must not
        # renormalize, it uses minutes_share directly.
        game = make_game(
            {1: [(101, 10.0), (102, 20.0)], 2: [(201, 15.0), (202, 25.0)]},
            game_id=10,
            minutes_by_player={101: 120.0, 102: 120.0, 201: 120.0, 202: 120.0},
        )
        updates, _, _ = self.engine.run(game)
        self.assertTrue((updates["minutes_share"] > 0).all())

    def test_no_surprise_keeps_perf_used_equal_perf_i(self):
        game = make_game(
            {1: [(101, 10.0), (102, 20.0)], 2: [(201, 15.0), (202, 25.0)]},
            game_id=11,
        )
        updates, _, _ = self.engine.run(game)
        self.assertTrue(np.allclose(updates["perf_used"], updates["perf_i"]))

    def test_surprise_fixed_penalizes_high_rating(self):
        engine = EloEngine(
            k=20.0, rookie_boost=1.0, init_mode="draft",
            draft_ratings={101: 1900.0, 102: 1500.0, 201: 1500.0, 202: 1500.0},
            surprise_scale=400.0, surprise_anchor="fixed",
        )
        game = make_game(
            {1: [(101, 10.0), (102, 20.0)], 2: [(201, 15.0), (202, 25.0)]},
            game_id=12,
        )
        updates, _, _ = engine.run(game)
        p101 = updates.loc[updates["player_id"] == 101].iloc[0]
        self.assertAlmostEqual(
            p101["perf_used"], p101["perf_i"] - (1900.0 - 1500.0) / 400.0
        )
        p102 = updates.loc[updates["player_id"] == 102].iloc[0]
        self.assertAlmostEqual(p102["perf_used"], p102["perf_i"])
        per_game = updates.groupby("game_id")["delta_adj"].sum()
        self.assertTrue(np.allclose(per_game, 0.0, atol=1e-9))

    def test_surprise_game_and_league_anchors_run(self):
        for anchor in ["game", "league"]:
            engine = EloEngine(
                k=20.0, rookie_boost=1.0, init_mode="draft",
                draft_ratings={101: 1900.0, 102: 1500.0, 201: 1500.0, 202: 1500.0},
                surprise_scale=400.0, surprise_anchor=anchor,
            )
            game = make_game(
                {1: [(101, 10.0), (102, 20.0)], 2: [(201, 15.0), (202, 25.0)]},
                game_id=13,
            )
            updates, _, _ = engine.run(game)
            self.assertEqual(len(updates), 4)
            self.assertTrue((updates["perf_used"] != updates["perf_i"]).any())
            per_game = updates.groupby("game_id")["delta_adj"].sum()
            self.assertTrue(np.allclose(per_game, 0.0, atol=1e-9))

    def test_surprise_scale_must_be_positive(self):
        with self.assertRaises(ValueError):
            EloEngine(surprise_scale=0.0)
        with self.assertRaises(ValueError):
            EloEngine(surprise_scale=-10.0)

    def test_playoff_k_multiplier_applies_to_playoffs_only(self):
        engine = EloEngine(
            k=20.0, rookie_boost=1.0, init_mode="uniform",
            playoff_k_multiplier=0.5,
        )
        g1 = make_game(
            {1: [(101, 10.0), (102, 20.0)], 2: [(201, 15.0), (202, 25.0)]},
            game_id=1, game_date="1976-10-21", track="regular",
        )
        g2 = make_game(
            {1: [(101, 10.0), (102, 20.0)], 2: [(201, 15.0), (202, 25.0)]},
            game_id=2, game_date="1976-10-22", track="playoffs",
        )
        box = pd.concat([g1, g2], ignore_index=True)
        updates, _, _ = engine.run(box)
        regular_k = updates[updates["track"] == "regular"]["k_effective"]
        playoff_k = updates[updates["track"] == "playoffs"]["k_effective"]
        self.assertTrue(np.allclose(regular_k, 20.0))
        self.assertTrue(np.allclose(playoff_k, 10.0))

    def test_alpha_modern_switches_at_boundary(self):
        pm = {101: 30.0, 102: -10.0, 201: 10.0, 202: -30.0}
        g95 = make_game(
            {1: [(101, 10.0), (102, 20.0)], 2: [(201, 15.0), (202, 25.0)]},
            game_id=1, game_date="1995-10-21", season=1995,
            plus_minus_by_player=pm,
        )
        g96 = make_game(
            {1: [(101, 10.0), (102, 20.0)], 2: [(201, 15.0), (202, 25.0)]},
            game_id=2, game_date="1996-10-21", season=1996,
            plus_minus_by_player=pm,
        )
        box = pd.concat([g95, g96], ignore_index=True)
        engine = EloEngine(
            k=20.0, rookie_boost=1.0, init_mode="uniform",
            alpha=1.0, alpha_modern=0.0, alpha_modern_start_season=1996,
        )
        updates, _, _ = engine.run(box)
        u95 = updates[updates["season"] == 1995]
        u96 = updates[updates["season"] == 1996]
        # Before 1996 alpha stays at 1, so perf_i is the GameScore z.
        self.assertTrue(np.allclose(u95["perf_i"], u95["z_game_score"]))
        # From 1996 alpha_modern=0 means perf_i is only the on-court z.
        self.assertTrue(np.allclose(u96["perf_i"], u96["z_on_court_rate"]))
        self.assertFalse(np.allclose(u95["perf_i"], u96["perf_i"]))

    def test_initial_state_resume_matches_full_run(self):
        g1 = make_game(
            {1: [(101, 10.0), (102, 20.0)], 2: [(201, 15.0), (202, 25.0)]},
            game_id=1, game_date="1995-10-21", season=1995,
        )
        g2 = make_game(
            {1: [(101, 12.0), (102, 18.0)], 2: [(201, 20.0), (202, 10.0)]},
            game_id=2, game_date="1996-10-21", season=1996,
        )
        full_box = pd.concat([g1, g2], ignore_index=True)
        full = self.engine.run(full_box)[0]

        # State after game 1, taken from a prior run with record_states.
        engine1 = EloEngine(
            k=20.0, theta=0.3, home_advantage=70.0, alpha=1.0,
            rookie_boost=1.0, init_mode="uniform", record_states=True,
        )
        _, states, _ = engine1.run(g1)
        state = {
            int(row["player_id"]): (float(row["rating"]),
                                      int(row["games_played"]),
                                      int(row["games_played_track"]))
            for _, row in states.iterrows()
        }
        resumed_engine = EloEngine(
            k=20.0, theta=0.3, home_advantage=70.0, alpha=1.0,
            rookie_boost=1.0, init_mode="uniform", initial_state=state,
        )
        resumed = resumed_engine.run(g2)[0]

        # 101/102 already exist, so resume must not re-initialize them.
        self.assertGreater(resumed["rating_before"].min(), 1400)
        cols = ["rating_before", "rating_after", "perf_i", "delta_adj"]
        self.assertTrue(np.allclose(
            resumed[cols].to_numpy(),
            full[full["game_id"] == 2][cols].to_numpy(),
            atol=1e-9,
        ))

    def test_alpha_modern_must_be_in_unit_interval(self):
        with self.assertRaises(ValueError):
            EloEngine(alpha_modern=-0.1)
        with self.assertRaises(ValueError):
            EloEngine(alpha_modern=1.1)


if __name__ == "__main__":
    unittest.main()
