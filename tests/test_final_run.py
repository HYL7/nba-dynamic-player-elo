"""Unit tests for final-run helpers (D-007 checkpoint reconstruction)."""

import sys
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_final_elo import checkpoint_from_updates, last_rating_map


class FinalRunTest(unittest.TestCase):
    def test_last_rating_map(self):
        updates = pd.DataFrame({
            "game_id": [1, 1, 2, 2],
            "game_date": ["1976-10-21", "1976-10-21", "1976-10-22", "1976-10-22"],
            "player_id": [101, 201, 101, 301],
            "rating_after": [1500.0, 1499.0, 1505.0, 1490.0],
        })
        ratings = last_rating_map(updates)
        self.assertEqual(ratings[101], 1505.0)
        self.assertEqual(ratings[201], 1499.0)
        self.assertEqual(ratings[301], 1490.0)

    def test_checkpoint_reconstruction(self):
        updates = pd.DataFrame({
            "game_id": [1, 1, 2, 2, 3, 3],
            "game_date": ["1976-10-21", "1976-10-21", "1976-10-22", "1976-10-22", "1977-10-20", "1977-10-20"],
            "season": [1976, 1976, 1976, 1976, 1977, 1977],
            "player_id": [101, 201, 101, 301, 101, 301],
            "rating_after": [1500.0, 1499.0, 1505.0, 1490.0, 1510.0, 1500.0],
        })
        state = checkpoint_from_updates(updates, 1976)
        self.assertEqual(state[101], (1505.0, 2, 2))
        self.assertEqual(state[201], (1499.0, 1, 1))
        self.assertEqual(state[301], (1490.0, 1, 1))
        self.assertNotIn(999, state)


if __name__ == "__main__":
    unittest.main()
