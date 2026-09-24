# Significance Layer Prototype

## Why the current neutral zone is limited

The current `neutral` class is a display-layer rule:

- `abs(regular_change_rate) < 0.00015`
- `abs(playoff_change_rate) < 0.00015`

This is useful for keeping the origin from looking artificially split, but it
has a structural weakness:

- the regular-season and playoff distributions do **not** have the same scale
- playoff rates are much more dispersed than regular-season rates

So the same absolute box around the origin does not mean the same thing on both
axes.

## Distribution mismatch

From the current sample:

- `regular_change_rate` std ≈ `0.000248`
- `playoff_change_rate` std ≈ `0.001430`

So the playoff axis is roughly `5.8x` more dispersed than the regular-season
axis. This is why a fixed "small circle" or fixed square around `(0,0)` is not
statistically symmetric.

## Better interpretation layer

Keep quadrants for directional interpretation, but define `neutral` using a
significance layer:

- `significant`: the player's playoff-vs-regular difference is distinguishable
  from zero
- `not significant`: no clear evidence of a stage difference

Then:

- `Q4 + significant` = strong playoff-penalty case
- `Q4 + not significant` = looks penalty-like, but evidence is weak
- `Q2 + significant` = strong playoff-reversal case
- near-origin `not significant` = true neutral

## Prototype test

Prototype uses a per-player permutation test on single-game relative Elo
changes:

- game metric: `delta_adj / rating_before`
- test statistic: `playoff_mean_relative_delta - regular_mean_relative_delta`
- permutation test for p-value
- bootstrap confidence interval as a second read

## Small sample prototype results

| Player | Reg G | PO G | Reg % | PO % | Effect % | Perm p | Significant |
|---|---:|---:|---:|---:|---:|---:|---|
| Billyray Bates | 187 | 6 | -0.0372 | 1.9320 | 1.9692 | 0.0002 | Yes |
| Michael Jordan | 1060 | 179 | 0.0102 | 0.1033 | 0.0931 | 0.0370 | Yes |
| LeBron James | 1621 | 305 | 0.0061 | 0.0772 | 0.0711 | 0.0324 | Yes |
| Dwyane Wade | 1054 | 177 | 0.0075 | 0.0583 | 0.0508 | 0.3247 | No |
| Tim Duncan | 1392 | 251 | 0.0066 | 0.0456 | 0.0390 | 0.2380 | No |
| Nikola Jokic | 809 | 100 | 0.0421 | 0.0810 | 0.0389 | 0.4041 | No |
| David Robinson | 987 | 123 | 0.0281 | -0.0683 | -0.0964 | 0.0614 | No |
| Joel Embiid | 490 | 67 | 0.1022 | -0.1878 | -0.2899 | 0.0004 | Yes |
| Austin Carr | 370 | 5 | 0.0104 | -0.9188 | -0.9292 | 0.0084 | Yes |

## Immediate takeaway

This looks promising:

- Jordan / LeBron come out as significantly more playoff-positive
- Embiid comes out as significantly playoff-negative
- Robinson looks penalty-like in direction, but is much weaker statistically

So the significance layer adds something real that the raw quadrant alone
cannot capture.

## Recommendation

Do **not** replace quadrants.

Instead, add a second layer:

1. Quadrant = direction of stage effect
2. Significance = confidence in that difference

This is a better long-term path than relying only on a fixed neutral box.
