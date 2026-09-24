"""Build instantaneous all-time No. 1 timeline for pk1.

Outputs to results/analysis/4.1_peak_elo:
- instant_no1_record_events_pk1.csv
- instant_no1_leader_changes_pk1.csv
- instant_no1_notes_pk1.md
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "results" / "nba_elo.db"
OUT_DIR = PROJECT_ROOT / "results" / "analysis" / "4.1_peak_elo"


QUERY = """
WITH base AS (
    SELECT
        re.game_id,
        re.game_date,
        re.player_id,
        re.team_id,
        p.full_name,
        re.home,
        re.rating_after,
        g.season_type,
        g.hometeam_name,
        g.awayteam_name,
        g.home_score,
        g.away_score,
        ROW_NUMBER() OVER (
            PARTITION BY re.game_id, re.player_id
            ORDER BY CASE re.source WHEN 'canonical' THEN 0 WHEN 'shared' THEN 1 ELSE 2 END
        ) AS rn
    FROM rating_events re
    JOIN players p
      ON p.player_id = re.player_id
    JOIN games g
      ON g.game_id = re.game_id
    WHERE re.playoff_k = 1.0
      AND re.source IN ('canonical', 'shared')
      AND g.is_usable = 1
),
uniq AS (
    SELECT
        game_id,
        game_date,
        player_id,
        team_id,
        full_name,
        home,
        rating_after,
        season_type,
        hometeam_name,
        awayteam_name,
        home_score,
        away_score
    FROM base
    WHERE rn = 1
),
rec AS (
    SELECT
        *,
        MAX(rating_after) OVER (
            ORDER BY game_date, game_id, player_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS prev_best
    FROM uniq
)
SELECT
    game_date,
    game_id,
    player_id,
    team_id,
    full_name,
    home,
    rating_after,
    prev_best,
    season_type,
    hometeam_name,
    awayteam_name,
    home_score,
    away_score
FROM rec
WHERE prev_best IS NULL OR rating_after > prev_best
ORDER BY game_date, game_id, player_id
"""


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rating_after"] = out["rating_after"].round(2)
    out["prev_best"] = out["prev_best"].round(2)
    out["opponent"] = out.apply(
        lambda r: r["awayteam_name"] if int(r["home"]) == 1 else r["hometeam_name"],
        axis=1,
    )
    out["result"] = out.apply(
        lambda r: (
            f"W {int(r['home_score'])}-{int(r['away_score'])}"
            if (int(r["home"]) == 1 and int(r["home_score"]) > int(r["away_score"]))
            or (int(r["home"]) == 0 and int(r["away_score"]) > int(r["home_score"]))
            else f"L {int(r['away_score'])}-{int(r['home_score'])}"
            if int(r["home"]) == 0
            else f"L {int(r['home_score'])}-{int(r['away_score'])}"
        ),
        axis=1,
    )
    out["game_type"] = out["season_type"].map({"Regular Season": "Regular", "Playoffs": "Playoffs"}).fillna(out["season_type"])
    return out[
        [
            "game_date",
            "full_name",
            "rating_after",
            "prev_best",
            "opponent",
            "result",
            "game_type",
            "game_id",
            "player_id",
            "team_id",
        ]
    ]


def build_notes(record_events: pd.DataFrame, leader_changes: pd.DataFrame) -> str:
    unique_holders = leader_changes["full_name"].tolist()
    stable_holders = [
        n for n in unique_holders
        if n not in {"Jim Ard", "Tom Boswell", "Dave Cowens", "John Havlicek", "Mike Newlin", "Doug Collins", "Kevin Porter", "Dan Issel"}
    ]
    return f"""# Instant No. 1 Timeline: pk1

## What this file set is

- `instant_no1_record_events_pk1.csv`: every time the all-time peak Elo record was raised
- `instant_no1_leader_changes_pk1.csv`: only the moments when the identity of No. 1 changed

Main world:

- `canonical pk1`

## Key counts

- record-raising events: {len(record_events)}
- leader-change events: {len(leader_changes)}

## Important interpretation note

The very first few entries in 1976 are startup-sensitive.

- everyone begins from the initialization world
- so the opening days can produce rapid one-game swaps
- these are real under the model, but not the best storytelling anchor for a final video section

That means the cleaner historical-holder list starts once the rating world has had time to separate.

## Historical-holder list

Raw leader-change names:

- {'; '.join(unique_holders)}

Cleaner video-facing holder list after ignoring the first startup burst:

- {'; '.join(stable_holders)}

## First-pass read

The timeline naturally splits into two stories:

1. startup turbulence in 1976
2. long-run historical handoffs:
   Kareem -> Moses -> Jordan -> Shaq -> LeBron

This is probably the cleanest narrative spine if the video wants to end on
"who actually stood at the top at any moment?"
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    raw = pd.read_sql_query(QUERY, con)
    con.close()

    record_events = add_context(raw)
    leader_changes = record_events[record_events["full_name"].ne(record_events["full_name"].shift(1))].reset_index(drop=True)

    record_events.to_csv(OUT_DIR / "instant_no1_record_events_pk1.csv", index=False)
    leader_changes.to_csv(OUT_DIR / "instant_no1_leader_changes_pk1.csv", index=False)
    (OUT_DIR / "instant_no1_notes_pk1.md").write_text(build_notes(record_events, leader_changes), encoding="utf-8")

    print(OUT_DIR / "instant_no1_record_events_pk1.csv")
    print(OUT_DIR / "instant_no1_leader_changes_pk1.csv")
    print(OUT_DIR / "instant_no1_notes_pk1.md")


if __name__ == "__main__":
    main()
