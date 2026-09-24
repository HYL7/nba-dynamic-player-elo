# 4.2 Video Outline

## Recommended role in the video

This works best as a short closing module, not a full standalone section.

Suggested rhythm:

1. explain the two axes in one sentence each
2. explain that color shows direction only
3. flash the four quadrants
4. pick a few extreme or familiar names
5. connect back to the peak-Elo section: playoff weighting changes not only peaks, but also the shape of a career

## Core method

Use the `canonical_pk2` worldline, with a stricter presentation gate:

- regular-season games >= 200
- playoff games >= 15

Axes:

- X axis = `regular_change_rate`
- Y axis = `playoff_change_rate`

Where:

- `regular_change_rate = regular_total_delta / regular_games / regular_baseline_elo`
- `playoff_change_rate = playoff_total_delta / playoff_games / playoff_baseline_elo`

Interpretation:

- right = tends to gain Elo in regular season
- left = tends to lose Elo in regular season
- up = tends to gain Elo in playoffs
- down = tends to lose Elo in playoffs

This is why the origin is `(0,0)`: it is the natural point where a player's Elo is, on average, not moving in either stage.

## What each quadrant means

- Q1: regular season up, playoffs up
- Q2: regular season down, playoffs up
- Q3: regular season down, playoffs down
- Q4: regular season up, playoffs down

Useful translation for viewers:

- Q1 = stable upward engines
- Q2 = playoff elevators
- Q3 = all-phase drifters
- Q4 = playoff drag / playoff penalty profile

## Why the video chart is quadrant-only

For the video version, the chart should stay quadrant-only.

Reason:

- the quadrant system describes career shape
- the old significance layer described playoff-vs-regular gap
- those are two different questions, so combining them in one chart makes the picture feel inconsistent

So the video chart should do one job well:

- show where a player sits in regular-vs-playoff Elo shape space

Any richer statistical extension can be mentioned briefly, but does not need to live on the main picture.

## Current limitation if we add significance back in

This is the key weakness of the current chart, and it should be stated plainly.

The two layers answer different questions:

- quadrant asks: what are the **signs** of regular and playoff change rate?
- significance asks: is the **difference between playoff and regular** large enough to separate from noise?

So they are not evaluating the same thing.

### Example of the mismatch

A player can be:

- clearly in `Q1`
- regular season positive
- playoffs positive

but still be `not significant`, if:

- both stages are positive
- and the gap between them is not large enough

That means `Q1 + not significant` does **not** mean "unclear player" in a general sense.
It only means "the playoff-vs-regular gap is unclear."

Likewise:

- a player can be strongly `Q4`
- yet what is statistically clear is not "playoffs are negative in absolute terms"
- but only "playoffs are worse than regular season"

This is why the current chart is useful as a first-pass comparison tool, but not a complete classification system.

## Long-term fix

If we want a logically unified system, the next version should separate three questions:

1. Is regular-season change rate positive, neutral, or negative?
2. Is playoff change rate positive, neutral, or negative?
3. Is the playoff-minus-regular gap itself significant or not?

That would give:

- a 9-class position system from the two axes
- plus an optional extra badge for whether the stage gap is statistically clear

This is cleaner than forcing a single `significant / not significant` layer to do two jobs at once.

## Best short-script structure

### Step 1: explain the graph fast

Possible English framing:

"The horizontal axis asks: does this player usually build Elo during the regular season, or bleed it? The vertical axis asks the same question for the playoffs. So every player gets a career-shaped fingerprint. And the filled points are the ones where the playoff-vs-regular split is actually clear enough to trust."

### Step 2: explain the four quadrants

Possible English framing:

"Top-right means a player tends to gain rating in both environments. Bottom-left means the opposite. Top-left is the really fun one: players whose Elo drifts in the regular season but climbs in the playoffs. Bottom-right is the danger zone: players who look healthier in the regular season than they do in postseason games."

### Step 3: show familiar anchors

Recommended highlighted names already on the chart:

- Michael Jordan
- Tim Duncan
- David Robinson
- Joel Embiid
- James Harden
- Baron Davis
- Reggie Miller
- Draymond Green

### Step 4: name additional recognizable examples

Good Q2, significant, more recognizable:

- Derrick Coleman
- Baron Davis
- Reggie Miller
- Draymond Green
- Dennis Johnson
- Maurice Cheeks
- Richard Hamilton
- Robert Horry
- Roy Hibbert

Good Q4, significant, more recognizable:

- Joel Embiid
- Julius Randle
- Kristaps Porzingis
- Brad Miller
- George McGinnis
- Keith Van Horn
- Kyle Kuzma

## Modern-player examples by quadrant

These are useful when you want the examples to feel more current for viewers.

### Q1: regular up, playoffs up

- LeBron James
- Nikola Jokic
- Giannis Antetokounmpo
- Jayson Tatum
- Jalen Brunson
- Luka Doncic
- Anthony Edwards

### Q2: regular down, playoffs up

- Draymond Green
- P.J. Tucker
- Rajon Rondo
- Khris Middleton
- Brook Lopez

Nearby modern playoff-positive shape:

- Jamal Murray can also be mentioned, but he lands in Q1 here rather than Q2

### Q3: regular down, playoffs down

- Rudy Gay

This is the sparsest modern-friendly quadrant in the current gated sample, so it is fine if the video does not force a long Q3 name list.

Important caution on `Q3`:

- this quadrant is very easy to fill with weak-role players, especially from earlier eras
- many of the farthest `Q3` names are low-impact or low-recognition players rather than strong historical anchors
- so `Q3` is the least presentation-friendly quadrant if the goal is recognizable examples

That is why:

- `Rudy Gay` is a more usable on-screen anchor than many of the raw distance leaders
- the video should not over-interpret the extreme `Q3` tail as a major basketball finding by itself

### Q4: regular up, playoffs down

- Joel Embiid
- Rudy Gobert
- Julius Randle
- Kristaps Porzingis
- Kyle Kuzma
- Andre Drummond
- DeMarcus Cousins
- James Harden
- Kevin Durant
- Damian Lillard
- Klay Thompson
- Karl-Anthony Towns
- Tyrese Haliburton
- Devin Booker

Important note on `Kevin Durant`:

- yes, he technically lands in `Q4` here
- but his playoff change rate is only slightly below zero, so he is very close to the horizontal axis
- the result is also clearly `not significant`
- in other words, this is better read as "KD is marginally on the Q4 side in this setup" rather than "KD is a strong playoff-penalty case"

## Composition caution by quadrant

Two quadrants have especially important sample-composition issues:

### Q1 can be inflated by rising modern stars

Many of the farthest modern `Q1` names are players whose careers are still in strong upward phases.

Examples currently near the outer `Q1` region:

- Ja Morant
- Paolo Banchero
- Jalen Brunson
- Jayson Tatum
- Jamal Murray
- Franz Wagner

This does not make the result wrong. But it does mean:

- distance from the origin is not automatically the same thing as historical certainty
- some extreme modern `Q1` cases may partly reflect incomplete-career growth curves

So for video use:

- modern names are great for intuitive examples
- but historical anchors like `Jordan`, `LeBron`, `Duncan`, and `Jokic` are usually safer interpretive anchors

### Q3 can be dominated by weak historical tail players

The opposite issue happens in `Q3`.

- players far from the origin there are often weak or low-usage players
- many come from older roster environments or low-impact career shapes
- that makes the quadrant less useful for star-based storytelling

So `Q3` is best treated as:

- a structural quadrant that exists in the map
- not a quadrant that necessarily needs heavy narrative emphasis in the main video

## Extra historical notes

Useful historical placements we checked:

- `Kareem Abdul-Jabbar` lands in `Q2` here, and not near the origin
- `Julius Erving` (`Dr. J`) technically lands in `Q4`, but extremely close to the origin, so he reads more like a near-neutral case than a strong Q4 anchor

## Future expansion direction: from 4 quadrants to 9 classes

This is a very reasonable next step, but it should stay as a brief future note in the main video, not as part of the main quadrant chart.

That is why your `KD` intuition is useful. In a future 9-class system, someone like `Kevin Durant` might be better described as:

- regular-season positive
- playoff neutral

rather than being forced into a hard `Q4` reading.

### Conceptual 9-class version

Cross the two stages independently:

- regular season: positive / neutral / negative
- playoffs: positive / neutral / negative

That gives 9 classes:

1. regular positive, playoff positive
2. regular positive, playoff neutral
3. regular positive, playoff negative
4. regular neutral, playoff positive
5. regular neutral, playoff neutral
6. regular neutral, playoff negative
7. regular negative, playoff positive
8. regular negative, playoff neutral
9. regular negative, playoff negative

This is probably closer to natural viewer language.

### What would be needed statistically

To make this version defensible, we would need to judge each axis separately:

- is regular-season change rate meaningfully above zero, below zero, or practically near zero?
- is playoff change rate meaningfully above zero, below zero, or practically near zero?

That is harder than the current setup, because:

- `not significant` is not the same thing as `neutral`
- true `neutral` usually needs an equivalence-style idea, not just failure to reject zero
- the regular-season and playoff distributions have different variance scales, so the same raw cutoff is not symmetric

### Recommended framing

The current 4-quadrant view is the right first-pass video tool.

The 9-class version is an excellent follow-up extension if we later want a more audience-friendly taxonomy, especially for edge cases like:

- `KD` style: regular positive, playoff near-neutral
- `modern Q4 stars` who are clearly worse in playoffs than regular season, but not always cleanly negative enough to deserve the same label strength

## How to talk about the named players

- `Jordan`: strong positive in both stages, and a natural top-right anchor
- `Duncan`: still Q1, useful as a reminder that not every great player needs a huge playoff-vs-regular split
- `Robinson`: directionally Q4, useful for discussing career shape rather than forcing a hard verdict
- `Embiid`: one of the clearest recognizable Q4 stars in the current map
- `Harden`: another modern Q4-style example
- `Baron Davis`: a clean recognizable Q2 case
- `Reggie Miller`: one of the best star-level Q2 examples
- `Draymond Green`: another strong Q2 case, and stylistically different from a pure scorer example

## Current gate full-sample axis extremes

These are not hand-picked names. They are the raw extrema from the current video gate:

- canonical `pk2`
- regular-season games >= 200
- playoff games >= 15

Four axis-direction extremes:

- biggest regular-season rise: `Jalen Johnson` (`regular_change_rate = +0.1229%`, `Q4`)
- biggest regular-season drop: `Anthony Roberts` (`regular_change_rate = -0.0778%`, `Q2`)
- biggest playoff rise: `Mickey Johnson` (`playoff_change_rate = +0.6222%`, `Q2`)
- biggest playoff drop: `Phil Ford` (`playoff_change_rate = -0.6519%`, `Q4`)

This can be a nice quick visual add-on if you want to show that the same map can surface "shape leaders" in different directions, not just famous stars.

## Editorial recommendation

For the main video, keep this section compact:

1. graph meaning
2. four quadrants
3. a few extreme or famous names
4. quick reminder that playoff weighting changes both peak rankings and career shape
5. if needed, briefly mention that future work could add a neutral zone / 9-class version for edge cases like KD

If you want to go deep on edge cases, neutral definitions, or a more statistical classification layer, that is better as a separate follow-up video.
