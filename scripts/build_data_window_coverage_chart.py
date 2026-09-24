"""Build the video-facing player-minute coverage chart for the data cutoff.

The chart answers a specific question: even when a historical game exists in
the game table, how many players in that game have usable (> 0) minute records?
The Dynamic Player Elo needs those player-minute records for within-game
performance comparison and minutes weighting.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data" / "Data2"
OUT = ROOT / "results" / "video" / "data_window"


def season_start_from_date(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    season = dates.dt.year.where(dates.dt.month >= 7, dates.dt.year - 1)
    # The 2019-20 bubble remained part of the 2019 season.
    season = season.where(~((dates.dt.year == 2020) & dates.dt.month.between(7, 10)), 2019)
    return season.astype("Int64")


def load_regular_games() -> pd.DataFrame:
    games = pd.read_csv(
        RAW / "Games.csv",
        usecols=[
            "gameId",
            "gameDateTimeEst",
            "gameType",
            "hometeamId",
            "awayteamId",
        ],
        low_memory=False,
    )
    games = games[games["gameType"].eq("Regular Season")].copy()
    games["season"] = season_start_from_date(games["gameDateTimeEst"])
    games = games.dropna(subset=["gameId", "season"]).drop_duplicates("gameId")
    games["gameId"] = games["gameId"].astype("int64")
    games["season"] = games["season"].astype(int)
    return games[["gameId", "season", "hometeamId", "awayteamId"]]


def count_players_with_minutes(game_ids: set[int]) -> pd.Series:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        RAW / "PlayerStatistics.csv",
        usecols=["gameId", "personId", "numMinutes"],
        chunksize=250_000,
        low_memory=False,
    ):
        chunk = chunk[chunk["gameId"].isin(game_ids)].copy()
        chunk["numMinutes"] = pd.to_numeric(chunk["numMinutes"], errors="coerce")
        chunk = chunk[chunk["numMinutes"].gt(0)]
        parts.append(chunk[["gameId", "personId"]].drop_duplicates())

    played = pd.concat(parts, ignore_index=True).drop_duplicates(["gameId", "personId"])
    return played.groupby("gameId").size().rename("players_with_minutes")


def build_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = load_regular_games()
    counts = count_players_with_minutes(set(games["gameId"].tolist()))
    game_level = games.merge(counts, how="left", left_on="gameId", right_index=True)
    game_level["players_with_minutes"] = game_level["players_with_minutes"].fillna(0).astype(int)

    summary = (
        game_level.groupby("season")["players_with_minutes"]
        .agg(
            game_count="size",
            avg_players="mean",
            median_players="median",
            p10_players=lambda x: x.quantile(0.10),
            p90_players=lambda x: x.quantile(0.90),
            pct_games_18plus=lambda x: 100.0 * x.ge(18).mean(),
        )
        .reset_index()
    )
    summary["season_label"] = summary["season"].map(
        lambda y: f"{y}-{str(y + 1)[-2:]}"
    )
    team_appearances = pd.concat(
        [
            games[["season", "hometeamId"]].rename(columns={"hometeamId": "team_id"}),
            games[["season", "awayteamId"]].rename(columns={"awayteamId": "team_id"}),
        ],
        ignore_index=True,
    ).dropna(subset=["team_id"])
    team_games = (
        team_appearances.groupby(["season", "team_id"])
        .size()
        .rename("games")
        .reset_index()
    )
    team_summary = (
        team_games.groupby("season")["games"]
        .agg(
            team_count="size",
            avg_games_per_team="mean",
            min_games_per_team="min",
            max_games_per_team="max",
        )
        .reset_index()
    )
    team_summary["season_label"] = team_summary["season"].map(
        lambda y: f"{y}-{str(y + 1)[-2:]}"
    )
    return summary, team_summary


def draw_line_chart(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    title: str,
    subtitle: str,
    output_path: Path,
    y_max: float,
    y_ticks: list[int],
    cutoff_value: float,
    sixties_value: float,
    modern_value: float,
    sixties_label: str,
    cutoff_label: str,
    modern_label: str,
    band_low: np.ndarray | None = None,
    band_high: np.ndarray | None = None,
    band_note: str | None = None,
) -> None:
    width, height = 1920, 1080
    image = Image.new("RGB", (width, height), "#08111f")
    draw = ImageDraw.Draw(image, "RGBA")
    font_regular = Path("C:/Windows/Fonts/arial.ttf")
    font_bold = Path("C:/Windows/Fonts/arialbd.ttf")

    def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        path = font_bold if bold else font_regular
        return ImageFont.truetype(str(path), size=size)

    left, top, right, bottom = 170, 235, 1825, 865
    xmin, xmax, ymin, ymax = 1953.5, 2026.5, 0.0, y_max

    def px(year: float) -> float:
        return left + (year - xmin) / (xmax - xmin) * (right - left)

    def py(value: float) -> float:
        return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

    for value in y_ticks:
        y = py(value)
        draw.line((left, y, right, y), fill=(120, 144, 165, 46), width=2)
        label = str(value)
        box = draw.textbbox((0, 0), label, font=font(23))
        draw.text((left - 24 - (box[2] - box[0]), y - 13), label, fill="#8fa8bc", font=font(23))

    if band_low is not None and band_high is not None:
        upper = [(px(x), py(y)) for x, y in zip(x_values, band_high)]
        lower = [(px(x), py(y)) for x, y in zip(x_values[::-1], band_low[::-1])]
        draw.polygon(upper + lower, fill=(83, 179, 255, 34))

    points = [(px(x), py(y)) for x, y in zip(x_values, y_values)]
    draw.line(points, fill="#65c7ff", width=7, joint="curve")

    cutoff = 1976
    cutoff_x, cutoff_y = px(cutoff), py(cutoff_value)
    dash = 14
    y = top
    while y < bottom:
        draw.line((cutoff_x, y, cutoff_x, min(y + dash, bottom)), fill="#ffb454", width=4)
        y += dash * 2
    draw.ellipse(
        (cutoff_x - 10, cutoff_y - 10, cutoff_x + 10, cutoff_y + 10),
        fill="#ffb454",
        outline="#08111f",
        width=3,
    )

    draw.text((left, 76), title, fill="#f4f8fc", font=font(50, bold=True))
    draw.text((left, 148), subtitle, fill="#9fb5c9", font=font(29))

    for year in [1954, 1960, 1970, 1976, 1980, 1990, 2000, 2010, 2020, 2025]:
        label = str(year)
        box = draw.textbbox((0, 0), label, font=font(22))
        color = "#ffbf6b" if year == 1976 else "#8fa8bc"
        draw.text((px(year) - (box[2] - box[0]) / 2, bottom + 24), label, fill=color, font=font(22))

    draw.text((left, 930), "Season start year", fill="#b7cadb", font=font(25))
    if band_note:
        draw.text((left, 1010), band_note, fill="#71889d", font=font(20))

    sixties_x, sixties_y = px(1964.5), py(sixties_value)
    sixties_text_y = min(sixties_value - y_max * 0.16, y_max * 0.72)
    sixties_text_y = max(sixties_text_y, y_max * 0.12)
    draw.line((sixties_x, sixties_y, px(1959), py(sixties_text_y + y_max * 0.04)), fill="#8aa2b8", width=3)
    draw.text((px(1954.8), py(sixties_text_y)), sixties_label, fill="#d9e7f5", font=font(26))

    cutoff_text_y = cutoff_value - y_max * 0.20
    draw.line((cutoff_x, cutoff_y, px(1982), py(cutoff_text_y + y_max * 0.02)), fill="#ffb454", width=3)
    draw.text(
        (px(1982.4), py(cutoff_text_y)),
        cutoff_label,
        fill="#ffd39a",
        font=font(27, bold=True),
        spacing=7,
    )
    modern_box = draw.textbbox((0, 0), modern_label, font=font(25))
    draw.text(
        (px(2003) - (modern_box[2] - modern_box[0]) / 2, py(min(modern_value + y_max * 0.10, y_max * 0.93))),
        modern_label,
        fill="#d9e7f5",
        font=font(25),
    )

    image.save(output_path)


def render(summary: pd.DataFrame, team_summary: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "player_minute_coverage_by_season.csv", index=False)
    team_summary.to_csv(OUT / "team_game_coverage_by_season.csv", index=False)

    plot = summary[summary["season"].between(1954, 2025)].copy()
    seasons = plot["season"].to_numpy(dtype=float)
    avg = plot["avg_players"].to_numpy(dtype=float)
    p10 = plot["p10_players"].to_numpy(dtype=float)
    p90 = plot["p90_players"].to_numpy(dtype=float)

    sixties = plot[plot["season"].between(1960, 1969)]
    sixties_weighted = np.average(sixties["avg_players"], weights=sixties["game_count"])
    cutoff_value = float(plot.loc[plot["season"].eq(1976), "avg_players"].iloc[0])
    modern = plot[plot["season"].between(1976, 2025)]
    modern_weighted = np.average(modern["avg_players"], weights=modern["game_count"])

    draw_line_chart(
        x_values=seasons,
        y_values=avg,
        title="Historical games exist — complete player-minute records do not",
        subtitle="Average number of players with recorded minutes per regular-season game",
        output_path=OUT / "player_minute_coverage_1954_2025.png",
        y_max=24,
        y_ticks=[0, 5, 10, 15, 20],
        cutoff_value=cutoff_value,
        sixties_value=sixties_weighted,
        modern_value=modern_weighted,
        sixties_label=f"1960s average: {sixties_weighted:.1f} players",
        cutoff_label=f"1976-77: {cutoff_value:.1f} players\nReliable player-level coverage begins",
        modern_label=f"1976-77 onward average: {modern_weighted:.1f}",
        band_low=p10,
        band_high=p90,
        band_note="Shaded band: 10th–90th percentile across games",
    )

    team_plot = team_summary[team_summary["season"].between(1954, 2025)].copy()
    team_sixties = team_plot[team_plot["season"].between(1960, 1969)]
    team_sixties_avg = np.average(
        team_sixties["avg_games_per_team"], weights=team_sixties["team_count"]
    )
    team_cutoff = float(
        team_plot.loc[team_plot["season"].eq(1976), "avg_games_per_team"].iloc[0]
    )
    full_length = team_plot[
        team_plot["season"].between(1976, 2025)
        & ~team_plot["season"].isin([1998, 2011, 2019, 2020])
    ]
    team_modern_avg = np.average(
        full_length["avg_games_per_team"], weights=full_length["team_count"]
    )
    draw_line_chart(
        x_values=team_plot["season"].to_numpy(dtype=float),
        y_values=team_plot["avg_games_per_team"].to_numpy(dtype=float),
        title="Game records are available much earlier",
        subtitle="Average number of recorded regular-season games per team",
        output_path=OUT / "team_game_coverage_1954_2025.png",
        y_max=90,
        y_ticks=[0, 20, 40, 60, 80],
        cutoff_value=team_cutoff,
        sixties_value=team_sixties_avg,
        modern_value=team_modern_avg,
        sixties_label=f"1960s average: {team_sixties_avg:.1f} games per team",
        cutoff_label=f"1976-77: {team_cutoff:.1f} games per team\nThe cutoff is not caused by missing games",
        modern_label=f"Full-length seasons after 1976 average: {team_modern_avg:.1f}",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "player_minute_coverage_by_season.csv", index=False)

    plot = summary[summary["season"].between(1954, 2025)].copy()
    seasons = plot["season"].to_numpy(dtype=float)
    avg = plot["avg_players"].to_numpy(dtype=float)
    p10 = plot["p10_players"].to_numpy(dtype=float)
    p90 = plot["p90_players"].to_numpy(dtype=float)

    sixties = plot[plot["season"].between(1960, 1969)]
    sixties_weighted = np.average(sixties["avg_players"], weights=sixties["game_count"])
    cutoff = 1976
    cutoff_value = float(plot.loc[plot["season"].eq(cutoff), "avg_players"].iloc[0])
    modern = plot[plot["season"].between(1976, 2025)]
    modern_weighted = np.average(modern["avg_players"], weights=modern["game_count"])



def main() -> None:
    summary, team_summary = build_summary()
    render(summary, team_summary)
    sixties = summary[summary["season"].between(1960, 1969)]
    sixties_avg = np.average(sixties["avg_players"], weights=sixties["game_count"])
    row_1976 = summary.loc[summary["season"].eq(1976)].iloc[0]
    print(f"1960s weighted average players/game: {sixties_avg:.3f}")
    print(
        "1976-77 average players/game: "
        f"{row_1976['avg_players']:.3f}; games >=18 players: "
        f"{row_1976['pct_games_18plus']:.2f}%"
    )
    print(OUT / "player_minute_coverage_1954_2025.png")
    print(OUT / "team_game_coverage_1954_2025.png")


if __name__ == "__main__":
    main()
