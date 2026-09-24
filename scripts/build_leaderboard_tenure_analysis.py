"""Build leaderboard tenure analysis for video section 4.4.

Focus:
- who was ever No. 1 on the live pk1 leaderboard
- how long each player stayed at No. 1
- how long each player stayed in the Top 10

Outputs to results/analysis/4.4_leaderboard_tenure:
- leaderboard_no1_duration_summary_pk1.csv
- leaderboard_ever_no1_by_first_date_pk1.csv
- leaderboard_no1_stints_pk1.csv
- leaderboard_top10_duration_summary_pk1.csv
- leaderboard_top10_stints_pk1.csv
- logic.md
- presentation_outline.md
- summary.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_PATH = PROJECT_ROOT / "results" / "leaderboard" / "top10_daily_canonical_pk1.csv"
OUT_DIR = PROJECT_ROOT / "results" / "analysis" / "4.4_leaderboard_tenure"


def build_no1_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    no1 = df[df["rank"] == 1].copy()
    no1["prev_player"] = no1["player_id"].shift(1)
    no1["new_stint"] = no1["player_id"].ne(no1["prev_player"]).astype(int)
    no1["stint_id"] = no1["new_stint"].cumsum()

    stints = no1.groupby(["player_id", "player_name", "stint_id"], as_index=False).agg(
        start_date=("frame_date", "min"),
        end_date=("frame_date", "max"),
        leaderboard_dates=("obs_days", "sum"),
        calendar_days=("span_days", "sum"),
    )

    summary = no1.groupby(["player_id", "player_name"], as_index=False).agg(
        first_no1_date=("frame_date", "min"),
        last_no1_date=("frame_date", "max"),
        no1_leaderboard_dates=("obs_days", "sum"),
        no1_calendar_days=("span_days", "sum"),
        no1_stints=("stint_id", "nunique"),
    )
    longest = stints.groupby(["player_id", "player_name"], as_index=False).agg(
        longest_no1_stint_days=("calendar_days", "max"),
        longest_no1_stint_dates=("leaderboard_dates", "max"),
    )
    summary = summary.merge(longest, on=["player_id", "player_name"], how="left")
    summary = summary.sort_values(
        ["no1_calendar_days", "no1_leaderboard_dates", "first_no1_date"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    stints = stints.sort_values(
        ["calendar_days", "leaderboard_dates", "start_date"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return summary, stints


def build_top10_tables(df: pd.DataFrame, previous_date_map: dict[pd.Timestamp, pd.Timestamp | pd.NaT]) -> tuple[pd.DataFrame, pd.DataFrame]:
    top10 = df.copy().sort_values(["player_id", "frame_date", "rank"]).reset_index(drop=True)
    top10["prev_player_date"] = top10.groupby("player_id")["frame_date"].shift(1)
    top10["expected_prev_date"] = top10["frame_date"].map(previous_date_map)
    top10["new_stint"] = top10["prev_player_date"].isna() | top10["prev_player_date"].ne(top10["expected_prev_date"])
    top10["new_stint"] = top10["new_stint"].astype(int)
    top10["stint_id"] = top10.groupby("player_id")["new_stint"].cumsum()

    stints = top10.groupby(["player_id", "player_name", "stint_id"], as_index=False).agg(
        start_date=("frame_date", "min"),
        end_date=("frame_date", "max"),
        leaderboard_dates=("obs_days", "sum"),
        calendar_days=("span_days", "sum"),
        best_rank=("rank", "min"),
    )

    summary = top10.groupby(["player_id", "player_name"], as_index=False).agg(
        first_top10_date=("frame_date", "min"),
        last_top10_date=("frame_date", "max"),
        top10_leaderboard_dates=("obs_days", "sum"),
        top10_calendar_days=("span_days", "sum"),
        top10_stints=("stint_id", "nunique"),
        best_rank=("rank", "min"),
    )
    longest = stints.groupby(["player_id", "player_name"], as_index=False).agg(
        longest_top10_stint_days=("calendar_days", "max"),
        longest_top10_stint_dates=("leaderboard_dates", "max"),
    )
    summary = summary.merge(longest, on=["player_id", "player_name"], how="left")
    summary = summary.sort_values(
        ["top10_calendar_days", "top10_leaderboard_dates", "first_top10_date"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    stints = stints.sort_values(
        ["calendar_days", "leaderboard_dates", "start_date"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return summary, stints


def md_table(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def build_logic() -> str:
    return """# 4.4 Leaderboard Tenure

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
"""


def build_presentation_outline(no1: pd.DataFrame, top10: pd.DataFrame) -> str:
    no1_names = ", ".join(no1["player_name"].head(5).tolist())
    top10_names = ", ".join(top10["player_name"].head(5).tolist())
    return f"""# 4.4 Presentation Outline

## Recommended role in the video

This works best as a late or final module.

Reason:

- `4.1` asks who reached the highest peak
- `4.4` asks who could actually stay at the top

That is a very natural contrast.

## Core framing

Possible English framing:

"Peak is one thing. But another question is: once a player reaches the top, how long can he actually live there? And even if he is not number one, how long can he stay inside the league's Elo top ten?"

## Main questions

1. Which players were ever No. 1?
2. Which players spent the longest total time at No. 1?
3. Which players had the longest single No. 1 reigns?
4. Which players stayed in the Top 10 the longest?

## Important caveat

This section should not be described as:

- all-time record-holder analysis
- peak refresh analysis

It is a live-leaderboard tenure analysis.

## Current strongest names

Top total No. 1 time currently starts with:

- {no1_names}

Top total Top-10 time currently starts with:

- {top10_names}

## Best short structure

1. define the metric in one sentence
2. show the list of everyone who was ever No. 1
3. show longest total time at No. 1
4. mention longest single reigns if useful
5. close with the Top-10 longevity list

## Why this section works

It gives you a strong final distinction:

- peak height
- peak ownership
- elite longevity

Those are related, but not the same.
"""


def build_summary(no1: pd.DataFrame, no1_stints: pd.DataFrame, top10: pd.DataFrame) -> str:
    holders = no1["player_name"].tolist()
    return f"""# 4.4 Leaderboard Tenure Summary

## Scope

- World: `canonical pk1`
- Source: `results/leaderboard/top10_daily_canonical_pk1.csv`
- This is a live leaderboard tenure module, not a peak-refresh module

## Players who were ever No. 1 ({len(holders)})

- {'; '.join(holders)}

## Longest total time at No. 1

{md_table(no1[['player_name', 'no1_calendar_days', 'no1_leaderboard_dates', 'no1_stints', 'longest_no1_stint_days']].head(15))}

## Longest single No. 1 reigns

{md_table(no1_stints[['player_name', 'start_date', 'end_date', 'calendar_days', 'leaderboard_dates']].head(15))}

## Longest total time in Top 10

{md_table(top10[['player_name', 'top10_calendar_days', 'top10_leaderboard_dates', 'top10_stints', 'best_rank', 'longest_top10_stint_days']].head(20))}
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_PATH, parse_dates=["frame_date"])
    df = df.sort_values(["frame_date", "rank", "player_id"]).reset_index(drop=True)

    dates = pd.Series(sorted(df["frame_date"].unique()))
    next_date_map = {dates.iloc[i]: (dates.iloc[i + 1] if i + 1 < len(dates) else pd.NaT) for i in range(len(dates))}
    prev_date_map = {dates.iloc[i]: (dates.iloc[i - 1] if i - 1 >= 0 else pd.NaT) for i in range(len(dates))}

    df["next_frame_date"] = df["frame_date"].map(next_date_map)
    df["span_days"] = (df["next_frame_date"] - df["frame_date"]).dt.days.fillna(0).astype(int)
    df["obs_days"] = 1

    no1_summary, no1_stints = build_no1_tables(df)
    top10_summary, top10_stints = build_top10_tables(df, prev_date_map)
    no1_first_date = no1_summary.sort_values(
        ["first_no1_date", "last_no1_date", "player_name"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    no1_first_date.insert(0, "first_no1_order", range(1, len(no1_first_date) + 1))

    no1_summary.to_csv(OUT_DIR / "leaderboard_no1_duration_summary_pk1.csv", index=False)
    no1_first_date.to_csv(OUT_DIR / "leaderboard_ever_no1_by_first_date_pk1.csv", index=False)
    no1_stints.to_csv(OUT_DIR / "leaderboard_no1_stints_pk1.csv", index=False)
    top10_summary.to_csv(OUT_DIR / "leaderboard_top10_duration_summary_pk1.csv", index=False)
    top10_stints.to_csv(OUT_DIR / "leaderboard_top10_stints_pk1.csv", index=False)
    (OUT_DIR / "logic.md").write_text(build_logic(), encoding="utf-8")
    (OUT_DIR / "presentation_outline.md").write_text(build_presentation_outline(no1_summary, top10_summary), encoding="utf-8")
    (OUT_DIR / "summary.md").write_text(build_summary(no1_summary, no1_stints, top10_summary), encoding="utf-8")

    print(OUT_DIR)


if __name__ == "__main__":
    main()
