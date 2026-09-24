# 4.2 Regular vs Playoff Shape

## Goal

Measure how a player's Elo tends to move in regular-season games versus
playoff games, using the full career sample instead of season-by-season
cuts that would discard too many players.

## Worldline

Main table uses `canonical_pk2`, so playoff games actively change Elo and
their impact is intentionally strong.

## Sample gates

- Regular-season games >= 100
- Playoff games >= 5

These are wide enough to keep broad coverage, but still remove the most
misleading tiny playoff samples.

## Core metrics

For each player:

- `regular_total_delta`: sum of `delta_adj` across all regular-season games
- `playoff_total_delta`: sum of `delta_adj` across all playoff games
- `regular_baseline_elo`: median pre-game Elo in regular-season games
- `playoff_baseline_elo`: median pre-game Elo in playoff games

Then define stage-level relative per-game change rates:

- `regular_change_rate = regular_total_delta / regular_games / regular_baseline_elo`
- `playoff_change_rate = playoff_total_delta / playoff_games / playoff_baseline_elo`

Interpretation:

- Positive: player's Elo tends to rise in that stage
- Negative: player's Elo tends to fall in that stage
- Larger magnitude: stage changes the player's Elo more aggressively

## Comparison signal

- `playoff_minus_regular = playoff_change_rate - regular_change_rate`

Interpretation:

- Positive: player tends to gain more Elo per playoff game than per regular-season game
- Negative: player tends to lose more, or gain less, in playoffs than in regular season

## Recommended 2D plot

- X axis: `regular_change_rate`
- Y axis: `playoff_change_rate`

Quadrants:

- Q1: rises in both stages
- Q2: falls in regular season, rises in playoffs
- Q3: falls in both
- Q4: rises in regular season, falls in playoffs

This is the main all-player mechanism view.

## Interpretation caveats by quadrant tail

The geometry of the chart is clean, but the tails are not equally clean in basketball terms.

### Q1 tail: often rising modern stars

In the stricter `r200_p15` presentation gate, some of the farthest `Q1` names are modern players whose careers are still climbing.

Useful examples we flagged:

- Ja Morant
- Paolo Banchero
- Jalen Brunson
- Jayson Tatum
- Jamal Murray
- Franz Wagner

Interpretation:

- these are often strong examples of playoff-positive shapes
- but they should not automatically be treated as settled all-time career fingerprints yet

### Q3 tail: often weak historical players

The `Q3` tail behaves very differently.

- many far-out `Q3` names are weak or low-recognition players
- this can make the quadrant look more interesting geometrically than it is narratively

So the video-facing approach should be:

- keep `Q3` in the chart
- use only one modest anchor if needed, such as `Rudy Gay`
- avoid overselling the extreme `Q3` tail as a major result

## Historical reference checks

We also checked a few older stars directly in the current `r200_p15` setup:

- `Kareem Abdul-Jabbar`: `Q2`, clearly away from the origin
- `Julius Erving`: technically `Q4`, but extremely close to the origin

This is useful because it shows that:

- not every great historical name becomes an extreme point
- and some famous older stars are better treated as anchor references than as outer-tail examples
