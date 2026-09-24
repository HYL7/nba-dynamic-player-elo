"""Build team-strength exploration tables from player Elo.

Main video-facing version:
- canonical pk1
- combine regular season + playoffs
- team-level pregame Elo from player pregame Elo weighted by actual game minutes

Outputs to results/analysis/4.3_team_strength_ideas:
- team_game_elo_pk1.csv
- player_season_elo_pk1.csv
- team_season_strength_pk1_all.csv
- team_season_strength_pk1_split.csv
- top_team_seasons_pk1.csv
- top_top2_seasons_pk1.csv
- top_top3_seasons_pk1.csv
- top_top5_seasons_pk1.csv
- top_seasonavg_top2_seasons_pk1.csv
- top_seasonavg_top3_seasons_pk1.csv
- top_seasonavg_top5_seasons_pk1.csv
- top_peak_top2_seasons_pk1.csv
- top_peak_top3_seasons_pk1.csv
- top_peak_top5_seasons_pk1.csv
- regular_vs_playoff_compare_pk1.csv
- lag_examples_pk1.csv
- rankings_table_pk1.csv
- summary.md
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
DB_PATH = RESULTS / "nba_elo.db"
OUT_DIR = RESULTS / "analysis" / "4.3_team_strength_ideas"
RANKINGS_PNG = OUT_DIR / "rankings_table_pk1.png"


BASE_SQL = """
WITH base AS (
    SELECT
        re.game_id,
        re.game_date,
        re.season,
        g.season_type,
        re.team_id,
        CASE
            WHEN re.team_id = g.hometeam_id THEN g.hometeam_name
            WHEN re.team_id = g.awayteam_id THEN g.awayteam_name
            ELSE CAST(re.team_id AS TEXT)
        END AS team_name,
        re.player_id,
        p.full_name,
        re.rating_before,
        re.minutes_share,
        ROW_NUMBER() OVER (
            PARTITION BY re.game_id, re.team_id
            ORDER BY re.minutes_share DESC, re.rating_before DESC, re.player_id
        ) AS rn
        ,
        ROW_NUMBER() OVER (
            PARTITION BY re.game_id, re.team_id
            ORDER BY re.rating_before DESC, re.minutes_share DESC, re.player_id
        ) AS rn_rating
    FROM rating_events re
    JOIN games g
      ON g.game_id = re.game_id
    LEFT JOIN players p
      ON p.player_id = re.player_id
    WHERE re.playoff_k = 1.0
      AND re.source IN ('shared', 'canonical')
      AND g.is_usable = 1
)
SELECT *
FROM base
ORDER BY game_date, game_id, team_name, full_name
"""


def season_label(season_start: int) -> str:
    return f"{season_start}-{str(season_start + 1)[-2:]}"


def table_text(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def fmt_entry(season_label: str, team_name: str, value: float) -> str:
    return f"{season_label} {team_name} ({value:.1f})"


def build_rankings_image(rankings: pd.DataFrame) -> None:
    width, height = 2500, 1700
    margin = 70
    col_w = 565
    row_h = 84
    header_h = 108
    bg = "#F7F3EA"
    panel = "#FBFAF6"
    ink = "#111827"
    muted = "#4B5563"
    line = "#D1D5DB"

    font_dir = Path("C:/Windows/Fonts")
    title_font = ImageFont.truetype(str(font_dir / "arialbd.ttf"), 52)
    head_font = ImageFont.truetype(str(font_dir / "arialbd.ttf"), 34)
    body_font = ImageFont.truetype(str(font_dir / "arial.ttf"), 28)
    rank_font = ImageFont.truetype(str(font_dir / "arialbd.ttf"), 30)

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    draw.text((margin, 28), "Team Elo Ranking Table: Canonical pk1", font=title_font, fill=ink)
    draw.text(
        (margin, 82),
        "Combined regular season + playoffs; rankings by average season strength",
        font=body_font,
        fill=muted,
    )
    top = 140
    left = margin
    table_w = width - margin * 2
    table_h = height - top - margin
    draw.rounded_rectangle([left, top, left + table_w, top + table_h], radius=18, fill=panel, outline="#C7C2B8", width=2)

    columns = [
        ("Rank", 120),
        ("Team Elo", col_w),
        ("Top-2", col_w),
        ("Top-3", col_w),
        ("Top-5", col_w),
    ]
    x = left + 20
    y = top + 18
    for label, w in columns:
        draw.text((x + 8, y + 16), label, font=head_font, fill=ink)
        x += w
    draw.line([left + 18, y + header_h, left + table_w - 18, y + header_h], fill=line, width=2)

    for i, row in rankings.iterrows():
        row_y = y + header_h + i * row_h
        if i > 0:
            draw.line([left + 18, row_y, left + table_w - 18, row_y], fill=line, width=1)
        x = left + 20
        draw.text((x + 12, row_y + 20), str(int(row["rank"])), font=rank_font, fill=ink)
        x += 120
        for key in ["team_entry", "top2_entry", "top3_entry", "top5_entry"]:
            draw.text((x + 8, row_y + 20), str(row[key]), font=body_font, fill=ink)
            x += col_w

    img.save(RANKINGS_PNG, quality=95)


def build_lag_examples(team_game: pd.DataFrame) -> pd.DataFrame:
    targets = [
        ("2003-04", "Lakers"),
        ("2010-11", "Heat"),
        ("2014-15", "Cavaliers"),
    ]
    rows = []
    for season_lbl, team_name in targets:
        sub = team_game[(team_game["season_label"] == season_lbl) & (team_game["team_name"] == team_name)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["game_date", "game_id"]).reset_index(drop=True)
        sub["game_no"] = sub.index + 1
        first10 = sub.head(10)
        last10 = sub.tail(10)
        peak_idx = sub["team_pregame_elo"].idxmax()
        peak_row = sub.loc[peak_idx]
        rows.append(
            {
                "season_label": season_lbl,
                "team_name": team_name,
                "games": len(sub),
                "peak_team_elo": float(sub["team_pregame_elo"].max()),
                "peak_date": peak_row["game_date"],
                "peak_game_no": int(peak_row["game_no"]),
                "first10_avg_team_elo": float(first10["team_pregame_elo"].mean()),
                "last10_avg_team_elo": float(last10["team_pregame_elo"].mean()),
                "last10_minus_first10_team_elo": float(last10["team_pregame_elo"].mean() - first10["team_pregame_elo"].mean()),
                "first10_avg_top2": float(first10["top2_weighted_elo"].mean()),
                "last10_avg_top2": float(last10["top2_weighted_elo"].mean()),
                "first10_avg_top5": float(first10["top5_weighted_elo"].mean()),
                "last10_avg_top5": float(last10["top5_weighted_elo"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_team_game(base: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["game_id", "game_date", "season", "season_type", "team_id", "team_name"]
    team_game = (
        base.assign(weighted_rating=base["rating_before"] * base["minutes_share"])
        .groupby(group_cols, as_index=False)
        .agg(team_pregame_elo=("weighted_rating", "sum"))
    )

    weighted = (
        base[base["rn"] <= 5]
        .assign(weighted_rating=lambda d: d["rating_before"] * d["minutes_share"])
        .groupby(group_cols + ["rn"], as_index=False)
        .agg(
            weighted_rating=("weighted_rating", "sum"),
            minutes_share=("minutes_share", "sum"),
        )
    )
    for n in [2, 3, 5]:
        sub = weighted[weighted["rn"] <= n].groupby(group_cols, as_index=False).agg(
            weighted_rating=("weighted_rating", "sum"),
            minutes_share=("minutes_share", "sum"),
        )
        team_game[f"top{n}_weighted_elo"] = sub["weighted_rating"] / sub["minutes_share"]
        team_game[f"top{n}_minutes_share"] = sub["minutes_share"]

    rating_rank = (
        base[base["rn_rating"] <= 5]
        .groupby(group_cols + ["rn_rating"], as_index=False)
        .agg(raw_rating=("rating_before", "mean"))
    )
    for n in [2, 3, 5]:
        sub = rating_rank[rating_rank["rn_rating"] <= n].groupby(group_cols, as_index=False).agg(
            instant_peak_rating=("raw_rating", "mean")
        )
        team_game[f"instant_top{n}_elo"] = sub["instant_peak_rating"]

    return team_game.sort_values(["game_date", "game_id", "team_name"]).reset_index(drop=True)


def build_player_season(base: pd.DataFrame) -> pd.DataFrame:
    player_season = (
        base.groupby(["season", "team_id", "team_name", "player_id", "full_name"], as_index=False)
        .agg(
            games=("game_id", "size"),
            avg_player_elo=("rating_before", "mean"),
            peak_player_elo=("rating_before", "max"),
            avg_minutes_share=("minutes_share", "mean"),
            total_minutes_share=("minutes_share", "sum"),
        )
    )
    return player_season


def build_seasonavg_core(player_season: pd.DataFrame, all_strength: pd.DataFrame, n: int) -> pd.DataFrame:
    qualified = player_season[player_season["games"] >= 20].copy()
    ranked = qualified.sort_values(
        ["season", "team_id", "avg_player_elo", "games", "total_minutes_share", "player_id"],
        ascending=[True, True, False, False, False, True],
    )
    core = (
        ranked.groupby(["season", "team_id", "team_name"], as_index=False)
        .head(n)
        .sort_values(["season", "team_id", "avg_player_elo"], ascending=[True, True, False])
    )
    summary = (
        core.groupby(["season", "team_id", "team_name"], as_index=False)
        .agg(
            **{
                f"seasonavg_top{n}_elo": ("avg_player_elo", "mean"),
                f"top{n}_players": ("full_name", lambda s: " / ".join(s.tolist())),
                f"top{n}_player_games": ("games", lambda s: " / ".join(str(int(v)) for v in s.tolist())),
            }
        )
    )
    summary = summary.merge(
        all_strength[["season", "team_id", "season_label", "games", "avg_team_elo"]],
        on=["season", "team_id"],
        how="left",
    )
    return summary.sort_values([f"seasonavg_top{n}_elo", "avg_team_elo"], ascending=[False, False]).reset_index(drop=True)


def build_peak_core(team_game: pd.DataFrame, n: int) -> pd.DataFrame:
    cols = ["season", "team_id", "team_name", "season_label", "game_date", f"instant_top{n}_elo"]
    peak_rows = (
        team_game[cols]
        .sort_values([f"instant_top{n}_elo", "game_date"], ascending=[False, True])
        .groupby(["season", "team_id", "team_name"], as_index=False)
        .head(1)
        .rename(columns={f"instant_top{n}_elo": f"peak_top{n}_elo", "game_date": f"peak_top{n}_date"})
    )
    peak_rows = peak_rows.merge(
        team_game.groupby(["season", "team_id"], as_index=False).agg(
            games=("game_id", "size"),
            avg_team_elo=("team_pregame_elo", "mean"),
        ),
        on=["season", "team_id"],
        how="left",
    )
    return peak_rows.sort_values([f"peak_top{n}_elo", "avg_team_elo"], ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    base = pd.read_sql_query(BASE_SQL, con)
    con.close()

    team_game = build_team_game(base)
    team_game["season_label"] = team_game["season"].map(season_label)
    team_game.to_csv(OUT_DIR / "team_game_elo_pk1.csv", index=False)
    player_season = build_player_season(base)
    player_season["season_label"] = player_season["season"].map(season_label)
    player_season.to_csv(OUT_DIR / "player_season_elo_pk1.csv", index=False)

    split_strength = (
        team_game.groupby(["season", "season_label", "season_type", "team_id", "team_name"], as_index=False)
        .agg(
            games=("game_id", "size"),
            avg_team_elo=("team_pregame_elo", "mean"),
            peak_team_elo=("team_pregame_elo", "max"),
            avg_top2_weighted_elo=("top2_weighted_elo", "mean"),
            avg_top2_share=("top2_minutes_share", "mean"),
            avg_top3_weighted_elo=("top3_weighted_elo", "mean"),
            avg_top3_share=("top3_minutes_share", "mean"),
            avg_top5_weighted_elo=("top5_weighted_elo", "mean"),
            avg_top5_share=("top5_minutes_share", "mean"),
            avg_instant_top2_elo=("instant_top2_elo", "mean"),
            avg_instant_top3_elo=("instant_top3_elo", "mean"),
            avg_instant_top5_elo=("instant_top5_elo", "mean"),
        )
    )
    split_strength.to_csv(OUT_DIR / "team_season_strength_pk1_split.csv", index=False)

    all_strength = (
        team_game.groupby(["season", "season_label", "team_id", "team_name"], as_index=False)
        .agg(
            games=("game_id", "size"),
            regular_games=("season_type", lambda s: int((s == "Regular Season").sum())),
            playoff_games=("season_type", lambda s: int((s == "Playoffs").sum())),
            avg_team_elo=("team_pregame_elo", "mean"),
            peak_team_elo=("team_pregame_elo", "max"),
            avg_top2_weighted_elo=("top2_weighted_elo", "mean"),
            avg_top2_share=("top2_minutes_share", "mean"),
            avg_top3_weighted_elo=("top3_weighted_elo", "mean"),
            avg_top3_share=("top3_minutes_share", "mean"),
            avg_top5_weighted_elo=("top5_weighted_elo", "mean"),
            avg_top5_share=("top5_minutes_share", "mean"),
            avg_instant_top2_elo=("instant_top2_elo", "mean"),
            avg_instant_top3_elo=("instant_top3_elo", "mean"),
            avg_instant_top5_elo=("instant_top5_elo", "mean"),
        )
    )
    all_strength = all_strength[all_strength["games"] >= 58].copy()
    all_strength.to_csv(OUT_DIR / "team_season_strength_pk1_all.csv", index=False)

    top_team = all_strength.sort_values(["avg_team_elo", "peak_team_elo"], ascending=[False, False]).head(20)
    top_top2 = all_strength.sort_values(["avg_top2_weighted_elo", "avg_team_elo"], ascending=[False, False]).head(20)
    top_top3 = all_strength.sort_values(["avg_top3_weighted_elo", "avg_team_elo"], ascending=[False, False]).head(20)
    top_top5 = all_strength.sort_values(["avg_top5_weighted_elo", "avg_team_elo"], ascending=[False, False]).head(20)

    top_team.to_csv(OUT_DIR / "top_team_seasons_pk1.csv", index=False)
    top_top2.to_csv(OUT_DIR / "top_top2_seasons_pk1.csv", index=False)
    top_top3.to_csv(OUT_DIR / "top_top3_seasons_pk1.csv", index=False)
    top_top5.to_csv(OUT_DIR / "top_top5_seasons_pk1.csv", index=False)

    seasonavg_top2 = build_seasonavg_core(player_season, all_strength, 2).head(20)
    seasonavg_top3 = build_seasonavg_core(player_season, all_strength, 3).head(20)
    seasonavg_top5 = build_seasonavg_core(player_season, all_strength, 5).head(20)
    peak_top2 = build_peak_core(team_game, 2).head(20)
    peak_top3 = build_peak_core(team_game, 3).head(20)
    peak_top5 = build_peak_core(team_game, 5).head(20)

    seasonavg_top2.to_csv(OUT_DIR / "top_seasonavg_top2_seasons_pk1.csv", index=False)
    seasonavg_top3.to_csv(OUT_DIR / "top_seasonavg_top3_seasons_pk1.csv", index=False)
    seasonavg_top5.to_csv(OUT_DIR / "top_seasonavg_top5_seasons_pk1.csv", index=False)
    peak_top2.to_csv(OUT_DIR / "top_peak_top2_seasons_pk1.csv", index=False)
    peak_top3.to_csv(OUT_DIR / "top_peak_top3_seasons_pk1.csv", index=False)
    peak_top5.to_csv(OUT_DIR / "top_peak_top5_seasons_pk1.csv", index=False)

    rankings = pd.DataFrame(
        {
            "rank": range(1, 16),
            "team_entry": [
                fmt_entry(r.season_label, r.team_name, r.avg_team_elo)
                for r in top_team.head(15).itertuples(index=False)
            ],
            "top2_entry": [
                fmt_entry(r.season_label, r.team_name, r.avg_top2_weighted_elo)
                for r in top_top2.head(15).itertuples(index=False)
            ],
            "top3_entry": [
                fmt_entry(r.season_label, r.team_name, r.avg_top3_weighted_elo)
                for r in top_top3.head(15).itertuples(index=False)
            ],
            "top5_entry": [
                fmt_entry(r.season_label, r.team_name, r.avg_top5_weighted_elo)
                for r in top_top5.head(15).itertuples(index=False)
            ],
        }
    )
    rankings.to_csv(OUT_DIR / "rankings_table_pk1.csv", index=False)
    build_rankings_image(rankings)

    regular = split_strength[(split_strength["season_type"] == "Regular Season") & (split_strength["games"] >= 50)].copy()
    playoffs = split_strength[(split_strength["season_type"] == "Playoffs") & (split_strength["games"] >= 8)].copy()
    regular["reg_rank"] = regular["avg_team_elo"].rank(method="dense", ascending=False)
    playoffs["po_rank"] = playoffs["avg_team_elo"].rank(method="dense", ascending=False)
    compare = regular.merge(
        playoffs[["season", "team_id", "team_name", "avg_team_elo", "po_rank"]],
        on=["season", "team_id", "team_name"],
        how="inner",
        suffixes=("_reg", "_po"),
    )
    compare["rank_gap"] = compare["reg_rank"] - compare["po_rank"]
    compare = compare.sort_values(["reg_rank", "po_rank"])
    compare.to_csv(OUT_DIR / "regular_vs_playoff_compare_pk1.csv", index=False)

    lag_examples = build_lag_examples(team_game)
    lag_examples.to_csv(OUT_DIR / "lag_examples_pk1.csv", index=False)

    summary = f"""# 4.3 Team Strength Exploration

## Main video-facing definition

Team strength is defined here as:

- `team_pregame_elo = sum(player_pregame_elo * minutes_share_in_that_game)`

This resolves the time-varying Elo issue naturally:

- use each player's Elo before that specific game
- weight by the minutes they actually played in that game

So this is a dynamic team-strength estimate, not a static roster estimate.

Main version below uses:

- `canonical pk1`
- regular season + playoffs combined

## Core add-on choices

- `Top-2` is the clean duo / star-pair lens
- `Top-5` is closer to actual team core strength

There are now three distinct ways to read `Top-N`:

- `weighted Top-N`: weight those players by the minutes they actually played
- `season-average Top-N`: rank a team's players by their average Elo over that season
- `instant-peak Top-N`: find the strongest single in-season moment for that team's top N players

So:

- `weighted` is better for on-court influence
- `season-average` is better for stable core-configuration framing
- `instant-peak` is better for pure "how scary was the ceiling at one moment" framing

## Highest combined team seasons by average team Elo

{table_text(top_team[['season_label', 'team_name', 'games', 'regular_games', 'playoff_games', 'avg_team_elo', 'peak_team_elo']])}

## Highest combined Top-2 seasons

{table_text(top_top2[['season_label', 'team_name', 'games', 'avg_top2_weighted_elo', 'avg_top2_share', 'avg_team_elo']])}

## Highest combined Top-3 seasons

{table_text(top_top3[['season_label', 'team_name', 'games', 'avg_top3_weighted_elo', 'avg_top3_share', 'avg_team_elo']])}

## Highest combined Top-5 seasons

{table_text(top_top5[['season_label', 'team_name', 'games', 'avg_top5_weighted_elo', 'avg_top5_share', 'avg_team_elo']])}

## Highest combined season-average Top-2 seasons

{table_text(seasonavg_top2[['season_label', 'team_name', 'games', 'seasonavg_top2_elo', 'top2_players', 'avg_team_elo']])}

## Highest combined season-average Top-3 seasons

{table_text(seasonavg_top3[['season_label', 'team_name', 'games', 'seasonavg_top3_elo', 'top3_players', 'avg_team_elo']])}

## Highest combined season-average Top-5 seasons

{table_text(seasonavg_top5[['season_label', 'team_name', 'games', 'seasonavg_top5_elo', 'top5_players', 'avg_team_elo']])}

## Highest combined instant-peak Top-2 seasons

{table_text(peak_top2[['season_label', 'team_name', 'games', 'peak_top2_elo', 'peak_top2_date', 'avg_team_elo']])}

## Highest combined instant-peak Top-3 seasons

{table_text(peak_top3[['season_label', 'team_name', 'games', 'peak_top3_elo', 'peak_top3_date', 'avg_team_elo']])}

## Highest combined instant-peak Top-5 seasons

{table_text(peak_top5[['season_label', 'team_name', 'games', 'peak_top5_elo', 'peak_top5_date', 'avg_team_elo']])}

## Regular vs playoff ranking check

This is only a diagnostic check, since the main video version combines both.

Top regular-season team seasons by average team Elo:

{table_text(regular.sort_values('avg_team_elo', ascending=False).head(10)[['season_label', 'team_name', 'games', 'avg_team_elo']])}

Top playoff team seasons by average team Elo:

{table_text(playoffs.sort_values('avg_team_elo', ascending=False).head(10)[['season_label', 'team_name', 'games', 'avg_team_elo']])}

## Lag-effect examples

These are useful for testing the idea that first-year superteams may inherit
very high player Elo from previous situations, then decline as the new roster
context suppresses individual box-score driven Elo accumulation.

{table_text(lag_examples)}
"""
    (OUT_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print(OUT_DIR / "summary.md")


if __name__ == "__main__":
    main()
