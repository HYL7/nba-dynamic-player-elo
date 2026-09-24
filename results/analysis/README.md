# Analysis Modules

This folder stores video-facing analysis outputs, organized by section/module.

## Modules

- `4.1_peak_elo`
  - Peak-Elo leaderboard material.
  - Includes the multi-`k` top-10 table, the `k=1` peak-match detail table, and related images.
  - Only keeps single-peak material; live-leaderboard tenure files were separated out into `4.4`.

- `4.2_regular_vs_playoff_shape`
  - All-player regular-vs-playoff Elo change-rate analysis.
  - Main output is the career-level stage-change table.
  - Also includes a `support_pk0_pk2_lift` subfolder for the earlier `pk0` vs `pk2` lift prototype.

- `4.3_team_strength_ideas`
  - Team-level aggregation ideas built from player Elo.
  - Includes weighted team strength, stable core-configuration tables, and related presentation files.

- `4.4_leaderboard_tenure`
  - Live leaderboard tenure analysis.
  - Focuses on who was ever No. 1, how long players stayed at No. 1, and how long they stayed inside the Top 10.
  - Also stores the supporting `instant_no1_*` files and leaderboard-timeline notes used for the final reflection layer.
