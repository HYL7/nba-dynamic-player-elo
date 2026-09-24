# 4.4 Leaderboard Tenure

## Goal

This section is not about peak-score refreshes.

It asks a different question:

- who actually lived at the top of the live Elo leaderboard?
- for how long?
- and who stayed inside the Top 10 for unusually long stretches?

## Core source

- `results/leaderboard/top10_daily_canonical_pk1.csv`

This is the daily live leaderboard used for the long-run ranking video.

So the interpretation is:

- `No. 1` = the player who sat first on that day's live Elo leaderboard
- `Top 10` = the players who were inside that day's live Elo top ten

This is different from:

- all-time record-holder analysis
- single highest peak analysis

## Duration logic

For each leaderboard date:

- rank 1 contributes one observed leaderboard date to that player
- the gap to the next leaderboard date contributes calendar duration

For Top 10:

- a player keeps the same stint only if he is still in the Top 10 on the next observed leaderboard date
- if he drops out and later returns, that begins a new stint

## Why this section is useful

This gets at a different kind of greatness:

- not just how high a player climbed
- but how long he could actually hold elite territory

That makes it a strong complement to peak-Elo material.
