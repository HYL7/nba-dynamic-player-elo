from pathlib import Path
import sqlite3

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "results" / "nba_elo.db"
OUT = ROOT / "results" / "analysis" / "4.3_era_elo_distribution"


def weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float:
    frame = pd.DataFrame({"value": values, "weight": weights}).dropna()
    frame = frame[frame["weight"] > 0].sort_values("value")
    cutoff = frame["weight"].sum() * q
    return float(frame.loc[frame["weight"].cumsum() >= cutoff, "value"].iloc[0])


def season_stats(group: pd.DataFrame) -> pd.Series:
    weights = group["minutes"].clip(lower=0)
    return pd.Series(
        {
            "players": len(group),
            "minutes_weighted_mean": (group["rating"] * weights).sum() / weights.sum(),
            "weighted_p10": weighted_quantile(group["rating"], weights, 0.10),
            "weighted_p25": weighted_quantile(group["rating"], weights, 0.25),
            "weighted_median": weighted_quantile(group["rating"], weights, 0.50),
            "weighted_p75": weighted_quantile(group["rating"], weights, 0.75),
            "weighted_p90": weighted_quantile(group["rating"], weights, 0.90),
            "unweighted_mean": group["rating"].mean(),
            "unweighted_median": group["rating"].median(),
            "unweighted_sd": group["rating"].std(ddof=0),
            "top_rating": group["rating"].max(),
        }
    )


def period_label(season: int) -> str:
    if season < 1980:
        return "1977-79"
    if season < 1990:
        return "1980s"
    if season < 2000:
        return "1990s"
    if season < 2010:
        return "2000s"
    if season < 2020:
        return "2010s"
    return "2020-26"


def draw_chart(stats: pd.DataFrame, path: Path) -> None:
    width, height = 1800, 1050
    img = Image.new("RGB", (width, height), "#FAFAF8")
    draw = ImageDraw.Draw(img)
    font_dir = Path("C:/Windows/Fonts")
    title = ImageFont.truetype(str(font_dir / "arialbd.ttf"), 46)
    subtitle = ImageFont.truetype(str(font_dir / "arial.ttf"), 24)
    body = ImageFont.truetype(str(font_dir / "arial.ttf"), 21)
    bold = ImageFont.truetype(str(font_dir / "arialbd.ttf"), 22)

    left, top, right, bottom = 125, 170, 1720, 910
    y_min, y_max = 1200.0, 2350.0
    x_min, x_max = float(stats["season"].min()), float(stats["season"].max())

    def xy(year: float, value: float) -> tuple[int, int]:
        x = left + (year - x_min) / (x_max - x_min) * (right - left)
        y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
        return int(x), int(y)

    draw.text((left, 45), "How the NBA Elo distribution changed over time", fill="#111827", font=title)
    draw.text(
        (left, 105),
        "Season-end canonical Elo, weighted by minutes played; every player contributes",
        fill="#4B5563",
        font=subtitle,
    )

    for val in [1200, 1400, 1600, 1800, 2000, 2200]:
        x0, y = xy(x_min, val)
        draw.line((left, y, right, y), fill="#D1D5DB", width=2)
        draw.text((48, y - 12), str(val), fill="#6B7280", font=body)
    for year in [1977, 1980, 1990, 2000, 2010, 2020, 2026]:
        x, _ = xy(year, y_min)
        draw.text((x - 25, bottom + 20), str(year), fill="#6B7280", font=body)

    upper90 = [xy(row.season, row.weighted_p90) for row in stats.itertuples()]
    lower10 = [xy(row.season, row.weighted_p10) for row in stats.iloc[::-1].itertuples()]
    draw.polygon(upper90 + lower10, fill="#DCE6F2")
    upper75 = [xy(row.season, row.weighted_p75) for row in stats.itertuples()]
    lower25 = [xy(row.season, row.weighted_p25) for row in stats.iloc[::-1].itertuples()]
    draw.polygon(upper75 + lower25, fill="#9DB9D5")

    median_pts = [xy(row.season, row.weighted_median) for row in stats.itertuples()]
    top_pts = [xy(row.season, row.top_rating) for row in stats.itertuples()]
    draw.line(median_pts, fill="#153A5B", width=5, joint="curve")
    draw.line(top_pts, fill="#B45309", width=4, joint="curve")

    legend_x, legend_y = 1180, 195
    draw.rectangle((legend_x, legend_y, legend_x + 34, legend_y + 18), fill="#DCE6F2")
    draw.text((legend_x + 48, legend_y - 5), "Middle 80% of player-minutes", fill="#374151", font=body)
    legend_y += 40
    draw.rectangle((legend_x, legend_y, legend_x + 34, legend_y + 18), fill="#9DB9D5")
    draw.text((legend_x + 48, legend_y - 5), "Middle 50% of player-minutes", fill="#374151", font=body)
    legend_y += 40
    draw.line((legend_x, legend_y + 8, legend_x + 34, legend_y + 8), fill="#153A5B", width=5)
    draw.text((legend_x + 48, legend_y - 5), "Minutes-weighted median", fill="#374151", font=body)
    legend_y += 40
    draw.line((legend_x, legend_y + 8, legend_x + 34, legend_y + 8), fill="#B45309", width=4)
    draw.text((legend_x + 48, legend_y - 5), "Highest season-end rating", fill="#374151", font=body)

    draw.text((left, 950), "Season ending year", fill="#4B5563", font=bold)
    img.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as con:
        data = pd.read_sql_query(
            """
            SELECT season, player_id, rating, games, minutes
            FROM season_ratings
            WHERE variant_id = 'canonical_pk1'
            """,
            con,
        )

    stats = data.groupby("season", as_index=False).apply(season_stats, include_groups=False)
    stats.to_csv(OUT / "season_end_distribution_canonical_pk1.csv", index=False)

    data["period"] = data["season"].map(period_label)
    period_rows = []
    for period, group in data.groupby("period", sort=False):
        row = season_stats(group).to_dict()
        row["period"] = period
        row["player_seasons"] = len(group)
        period_rows.append(row)
    periods = pd.DataFrame(period_rows)
    periods["weighted_p90_p10_gap"] = periods["weighted_p90"] - periods["weighted_p10"]
    periods["top_minus_weighted_median"] = periods["top_rating"] - periods["weighted_median"]
    cols = ["period", "player_seasons"] + [c for c in periods.columns if c not in {"period", "player_seasons"}]
    periods = periods[cols]
    periods.to_csv(OUT / "period_distribution_canonical_pk1.csv", index=False)

    draw_chart(stats, OUT / "season_end_elo_distribution_canonical_pk1.png")
    print(periods.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
