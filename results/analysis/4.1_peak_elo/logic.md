# 4.1 Peak Elo

## Goal

Summarize the most intuitive final-result layer of the project: each player's
single highest instant Elo point.

## Core tables

- `peak_elo_top10_by_playoff_k.csv`
  - Top 10 peak-Elo list under `k = 0 / 0.5 / 1 / 1.5 / 2`
  - One row per rank, one column per playoff-weight setting

- `peak_elo_top10_k1_details.csv`
  - Detailed match context for the `k=1` top 10
  - Includes peak Elo, date, opponent, result, game-type label, and compact stat line

## Interpretation

- `k=0` is the pure regular-season worldline: playoff games exist on the
  timeline, but they do not change Elo.
- Higher `k` values increase the influence of playoff games.
- Comparing columns answers a simple question:
  which players keep the same historical peak, and which players get
  meaningfully re-ranked once playoffs matter more?

## Why this section matters

This module is the cleanest “result reveal” layer for the video:

- who is No. 1?
- who moves up or down as playoff weight increases?
- whose peak switches from regular season to playoffs?

It is intentionally peak-focused and should later be paired with `4.2` so the
video does not over-interpret one isolated high point.
